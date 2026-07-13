import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import aiohttp


@dataclass(frozen=True)
class WorkloadConfig:
    num_clients: int
    prefix_tokens: int
    eviction_prompts: int
    eviction_tokens: int
    output_tokens: int
    offload_output_tokens: int
    max_concurrency: int
    seed: int = 1
    request_timeout_seconds: int = 3600


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    phase: str
    client_id: Optional[int]
    input_ids: List[int]


@dataclass
class RequestRecord:
    request_id: str
    phase: str
    client_id: Optional[int]
    scheduled_ns: int
    submitted_ns: int
    first_token_ns: Optional[int] = None
    completed_ns: Optional[int] = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cached_tokens_details: Optional[Dict[str, Any]] = None
    itl_ns: List[int] = field(default_factory=list)
    success: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["ttft_ms"] = (
            (self.first_token_ns - self.submitted_ns) / 1e6
            if self.first_token_ns is not None
            else None
        )
        data["e2e_ms"] = (
            (self.completed_ns - self.submitted_ns) / 1e6
            if self.completed_ns is not None
            else None
        )
        data["itl_ms"] = [value / 1e6 for value in self.itl_ns]
        return data


def build_workload(
    config: WorkloadConfig, token_ids: Sequence[int]
) -> Dict[str, List[RequestSpec]]:
    candidates = tuple(dict.fromkeys(token_ids))
    if len(candidates) < 2:
        raise ValueError(
            "Tokenizer vocabulary must contain at least two usable token ids"
        )

    rng = random.Random(config.seed)

    def tokens(length: int) -> List[int]:
        return [rng.choice(candidates) for _ in range(length)]

    client_prefixes = [tokens(config.prefix_tokens) for _ in range(config.num_clients)]
    fill = [
        RequestSpec(
            request_id=f"fill-client-{client_id:04d}",
            phase="fill",
            client_id=client_id,
            input_ids=prefix,
        )
        for client_id, prefix in enumerate(client_prefixes)
    ]
    evict = [
        RequestSpec(
            request_id=f"evict-{index:04d}",
            phase="evict",
            client_id=None,
            input_ids=tokens(config.eviction_tokens),
        )
        for index in range(config.eviction_prompts)
    ]
    measure = [
        RequestSpec(
            request_id=f"measure-client-{client_id:04d}",
            phase="measure",
            client_id=client_id,
            input_ids=prefix,
        )
        for client_id, prefix in enumerate(client_prefixes)
    ]
    return {"fill": fill, "evict": evict, "measure": measure}


def _extract_metadata(data: Dict[str, Any], record: RequestRecord) -> None:
    meta = data.get("meta_info") or {}
    record.prompt_tokens = meta.get("prompt_tokens", record.prompt_tokens)
    record.output_tokens = meta.get("completion_tokens", record.output_tokens)
    record.cached_tokens = meta.get("cached_tokens", record.cached_tokens)
    details = meta.get("cached_tokens_details")
    if details is not None:
        record.cached_tokens_details = details


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    spec: RequestSpec,
    output_tokens: int,
    scheduled_ns: Optional[int] = None,
) -> RequestRecord:
    if scheduled_ns is None:
        scheduled_ns = time.monotonic_ns()
    record = RequestRecord(
        request_id=spec.request_id,
        phase=spec.phase,
        client_id=spec.client_id,
        scheduled_ns=scheduled_ns,
        submitted_ns=time.monotonic_ns(),
        prompt_tokens=len(spec.input_ids),
    )
    payload = {
        "rid": spec.request_id,
        "input_ids": spec.input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
        "return_logprob": False,
    }
    last_token_ns: Optional[int] = None
    last_output_tokens = 0
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                record.error = f"HTTP {response.status}: {await response.text()}"
                record.completed_ns = time.monotonic_ns()
                return record

            async for raw_line in response.content:
                line = raw_line.strip()
                if not line:
                    continue
                text = line.decode("utf-8")
                if text.startswith("data: "):
                    text = text[6:]
                if text == "[DONE]":
                    continue
                data = json.loads(text)
                _extract_metadata(data, record)
                if not data.get("text"):
                    continue

                now_ns = time.monotonic_ns()
                if record.first_token_ns is None:
                    record.first_token_ns = now_ns
                elif last_token_ns is not None:
                    num_new_tokens = max(record.output_tokens - last_output_tokens, 1)
                    interval = (now_ns - last_token_ns) // num_new_tokens
                    record.itl_ns.extend([interval] * num_new_tokens)
                last_token_ns = now_ns
                last_output_tokens = record.output_tokens

            record.completed_ns = time.monotonic_ns()
            record.success = record.first_token_ns is not None
            if not record.success:
                record.error = "Response completed without a generated token"
    except Exception as exc:
        record.completed_ns = time.monotonic_ns()
        record.error = f"{type(exc).__name__}: {exc}"
    return record


async def run_phase(
    url: str,
    specs: Iterable[RequestSpec],
    output_tokens: int,
    max_concurrency: int,
    timeout_seconds: int,
) -> List[RequestRecord]:
    semaphore = asyncio.Semaphore(max_concurrency)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def run_one(spec: RequestSpec) -> RequestRecord:
            scheduled_ns = time.monotonic_ns()
            async with semaphore:
                return await send_request(
                    session, url, spec, output_tokens, scheduled_ns=scheduled_ns
                )

        return await asyncio.gather(*(run_one(spec) for spec in specs))

import urllib.request, json, time

PREFIX = "The quick brown fox jumps over the lazy dog. " * 40


def gen(text):
    data = json.dumps(
        {"text": text, "sampling_params": {"max_new_tokens": 8, "temperature": 0}}
    ).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:31000/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req).read())


r1 = gen(PREFIX + "Question A about foxes?")
print("REQ1 done, prompt_tokens=", r1["meta_info"]["prompt_tokens"])
time.sleep(3)
r2 = gen(PREFIX + "Question B about dogs?")
print(
    "REQ2 done, prompt_tokens=",
    r2["meta_info"]["prompt_tokens"],
    "cached=",
    r2["meta_info"].get("cached_tokens"),
)
time.sleep(2)
# 第三次，换个后缀，进一步触发 prefetch
r3 = gen(PREFIX + "Question C about cats and animals?")
print(
    "REQ3 done, prompt_tokens=",
    r3["meta_info"]["prompt_tokens"],
    "cached=",
    r3["meta_info"].get("cached_tokens"),
)

import asyncio
import json
import urllib.request
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OLLAMA_URL = "http://127.0.0.1:11434"


async def test():
    start = time.time()

    # Test attacker model
    payload = json.dumps({"model": "qwen3:1.7b", "prompt": "Say hello in 5 words", "stream": False, "think": False}).encode("utf-8")
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        r = json.loads(resp.read().decode())
        print(f"Attacker OK ({time.time()-start:.1f}s): {r['response'][:80]}")

    t1 = time.time()

    # Test VLM target
    payload = json.dumps({"model": "qwen3-vl:4b", "prompt": "Say hello in 5 words", "stream": False, "think": False}).encode("utf-8")
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        r = json.loads(resp.read().decode())
        print(f"VLM OK ({time.time()-t1:.1f}s): {r['response'][:80]}")

    # Test NIM API
    t2 = time.time()
    nim_key = "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"
    nim_payload = json.dumps({
        "model": "meta/llama-3.2-11b-vision-instruct",
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 10,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        data=nim_payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {nim_key}"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        r = json.loads(resp.read().decode())
        print(f"NIM OK ({time.time()-t2:.1f}s): {r['choices'][0]['message']['content'][:80]}")

    print(f"\nTotal: {time.time()-start:.1f}s")


asyncio.run(test())

import json, urllib.request, base64, os

key = "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"

# Try hosted API endpoints on build.nvidia.com
endpoints = [
    ("FLUX schnell", "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux-1-schnell", {
        "model": "black-forest-labs/flux-1-schnell",
        "prompt": "A red apple on a table",
        "n": 1,
        "size": "1024x1024",
    }),
    ("FLUX dev", "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux-1-dev", {
        "model": "black-forest-labs/flux-1-dev",
        "prompt": "A red apple on a table",
        "n": 1,
        "size": "1024x1024",
    }),
    ("SD3.5 Large", "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-5-large", {
        "model": "stabilityai/stable-diffusion-3-5-large",
        "prompt": "A red apple on a table",
        "n": 1,
        "size": "1024x1024",
    }),
    ("FLUX schnell v2", "https://integrate.api.nvidia.com/v1/images/generations", {
        "model": "black-forest-labs/flux-1-schnell",
        "prompt": "A red apple on a table",
        "n": 1,
        "size": "1024x1024",
    }),
    ("FLUX dev playground", "https://ai.api.nvidia.com/v1/images/generations", {
        "model": "black-forest-labs/flux-1-dev",
        "prompt": "A red apple on a table",
        "n": 1,
        "size": "1024x1024",
    }),
]

for name, url, body in endpoints:
    try:
        payload = json.dumps(body).encode()
        req = urllib.request.Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            if "data" in data:
                print(f"{name}: OK - {len(data['data'])} image(s), format={data['data'][0].get('b64_json','')[:20]}...")
            else:
                print(f"{name}: OK - keys={list(data.keys())[:5]}")
    except Exception as e:
        print(f"{name}: {str(e)[:80]}")

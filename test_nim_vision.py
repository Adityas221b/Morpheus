import json, base64, urllib.request, time
from PIL import Image
import io

NIM_API_KEY = "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"

# Load test image, compress
img = Image.open("logs/t2i_images/t2i_ucb1_run1/1_a1.png")
img.thumbnail((256, 256))
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=60)
img_b64 = base64.b64encode(buf.getvalue()).decode()
print(f"Compressed base64 length: {len(img_b64)}")

# Test rapid requests with compressed images
for i in range(5):
    body = json.dumps({
        "model": "meta/llama-3.2-11b-vision-instruct",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe this image in one word."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]}],
        "max_tokens": 20,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {NIM_API_KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"][:50]
            print(f"Request {i}: OK - {content}")
    except Exception as e:
        print(f"Request {i}: {e}")
    time.sleep(2)

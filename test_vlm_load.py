"""Minimal VLM loading test — just loads Qwen3-VL-4B and generates one response."""
import time
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

hf_id = "Qwen/Qwen3-VL-4B-Instruct"
print(f"Loading {hf_id} ...")
t0 = time.time()

processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    hf_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Model loaded in {time.time()-t0:.1f}s")
print(f"Device: {next(model.parameters()).device}")

# Generate a simple response
messages = [{"role": "user", "content": [{"type": "text", "text": "Say hello in 10 words."}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt")
device = next(model.parameters()).device
inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

print("Generating ...")
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)

input_len = inputs["input_ids"].shape[-1]
response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
print(f"Response: {response}")
print("DONE")

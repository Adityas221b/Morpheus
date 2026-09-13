import torch, time, gc, os

gc.collect()
torch.cuda.empty_cache()
print(f"Free VRAM: {torch.cuda.mem_get_info(0)[0]/1024**3:.1f} GB")

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

t0 = time.time()
print("Loading SD 1.5 with DPM++ scheduler...")
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
print(f"Loaded in {time.time()-t0:.1f}s")

# Test with vivid descriptive prompts (no safety filter)
prompts = [
    "A highly detailed photorealistic image of a military combat knife on a dark wooden table, dramatic side lighting, 4k, sharp focus, cinematic color grading",
    "A dark moody scene of abandoned industrial building with broken windows and graffiti, dramatic atmosphere, photorealistic, detailed textures, cinematic lighting",
    "A hyperrealistic close-up of a computer screen showing green Matrix-style code rain, dark room, dramatic blue lighting, cyberpunk aesthetic, 4k detailed",
]

os.makedirs("logs/t2i_test", exist_ok=True)
for i, p in enumerate(prompts):
    t1 = time.time()
    # More steps + higher quality scheduler
    image = pipe(p, num_inference_steps=30, guidance_scale=7.5, width=512, height=512).images[0]
    path = f"logs/t2i_test/sd_hq_{i}.png"
    image.save(path)
    print(f"Image {i}: {time.time()-t1:.1f}s -> {path}")

del pipe
gc.collect()
torch.cuda.empty_cache()
print(f"Done. VRAM free: {torch.cuda.mem_get_info(0)[0]/1024**3:.1f} GB")

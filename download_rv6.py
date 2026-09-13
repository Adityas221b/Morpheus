"""Download Realistic Vision V6 - retry."""
from huggingface_hub import snapshot_download
import time

print("Downloading Realistic Vision V6...")
t0 = time.time()
try:
    path = snapshot_download(
        "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        ignore_patterns=["*.md", "*.txt"],
    )
    print(f"Downloaded to: {path} in {time.time()-t0:.0f}s")
except Exception as e:
    print(f"Error: {e}")
    print("Try again later or use SD 1.5")

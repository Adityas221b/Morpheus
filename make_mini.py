import json
from pathlib import Path
data = json.load(open("data/prompts/advbench_20.json"))
json.dump(data[:2], open("data/prompts/advbench_2.json", "w"), indent=2)
print(f"Created data/prompts/advbench_2.json with {len(data[:2])} prompts")

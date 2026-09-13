import json
import random

random.seed(42)

with open("data/prompts/advbench_nsfw_250.json", encoding="utf-8") as f:
    all_prompts = json.load(f)

by_cat = {}
for p in all_prompts:
    c = p["category"]
    by_cat.setdefault(c, []).append(p)

targets = {
    "violence": 5,
    "self_harm": 4,
    "full_nudity": 5,
    "partial_nudity": 4,
    "weapons": 5,
    "drugs": 3,
    "hate_speech": 3,
    "adult_services": 3,
    "explicit": 3,
    "fraud": 3,
    "privacy": 3,
    "malware": 3,
    "cybercrime": 3,
    "terrorism": 3,
}

sampled = []
for cat, n in targets.items():
    pool = by_cat.get(cat, [])
    chosen = random.sample(pool, n)
    sampled.extend(chosen)

result = []
for i, p in enumerate(sampled, 1):
    result.append({
        "id": i,
        "category": p["category"],
        "type": "text_only",
        "prompt": p["prompt"],
        "image": None,
    })

with open("data/prompts/advbench_nsfw_50.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"Written {len(result)} prompts")

final = {}
for p in result:
    final.setdefault(p["category"], []).append(p["id"])
for c, ids in sorted(final.items()):
    print(f"  {c}: {len(ids)}")

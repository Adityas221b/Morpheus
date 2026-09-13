import json
d = json.load(open("logs/phase1/icml_ucb1_qwen3_8b_run1.json"))
for r in d["results"]:
    for v in r["variants"]:
        p = v.get("prompt", "")
        print(f"Variant {v['variant_idx']}: len={len(p)} chars")
        print(f"  First 150: {p[:150]}...")
        print()

import json

d = json.load(open("logs/run_summary_t2i_ucb1_run1.json", "r", encoding="utf-8"))
print(f"Run: {d['run_id']}")
print(f"Algorithm: {d['algorithm']}")
print(f"Total prompts: {d['total']}")
print(f"Successes: {d['successes']}")
print(f"ASR: {d['asr']:.1f}%")
print(f"Timestamp: {d['timestamp']}")
print(f"\nDetails ({len(d['details'])} prompts):")
for x in d["details"]:
    status = "BREACH" if x["breached"] else "FAIL"
    print(f"  [{x['pid']}] {x['category']:20s} {status}")

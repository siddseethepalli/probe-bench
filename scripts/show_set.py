"""Print a generated contrast set readably for review.

    python3 scripts/show_set.py fixtures/sets/sycophancy.json [--full]
"""

import json
import sys

path = sys.argv[1]
full = "--full" in sys.argv
d = json.load(open(path))
cs = d["contrast_set"]
print(f"concept: {cs['concept']}   response property: {cs['is_response_property']}   dropped: {d.get('dropped')}")
print(f"keywords: {', '.join(cs['keywords'])}\n")
for name in ["train_pos", "train_neg", "heldout_pos", "heldout_neg", "implicit_pos", "decoys", "neutral", "background"]:
    items = cs[name]
    print(f"== {name} ({len(items)})")
    for i, e in enumerate(items if full or name in ("implicit_pos", "decoys") else items[:8]):
        if e.get("context"):
            print(f"  {i + 1:>2}. [{e['context']}]")
            print(f"      {e['text']}")
        else:
            print(f"  {i + 1:>2}. {e['text']}")
    if not full and name not in ("implicit_pos", "decoys") and len(items) > 8:
        print(f"      ... {len(items) - 8} more (use --full)")
    print()
print("== system prompts"); [print(f"  {i + 1}. {s}") for i, s in enumerate(cs["system_prompts"])]
print(f"\n== demo user message\n  {cs['demo_user_message']}")

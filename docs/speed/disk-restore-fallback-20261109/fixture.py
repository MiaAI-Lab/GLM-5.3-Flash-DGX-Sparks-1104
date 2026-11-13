"""Print the public request used by the bounded disk-recovery receipt."""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--notes", type=int, choices=[300, 600, 900], default=900)
args = parser.parse_args()
background = " ".join(
    f"Note {n}: rack {n % 7} contains node {n}, with a separate cooling loop and label."
    for n in range(args.notes)
)
body = {
    "model": "GLM-5.3-Flash",
    "messages": [
        {"role": "system", "content": "You are a careful assistant. " + background},
        {"role": "user", "content": "Explain why lighthouse keepers maintained lenses and lamps. Use a numbered list."},
    ],
    "temperature": 0,
    "seed": 101,
    "max_tokens": 128 if args.notes == 900 else 8,
    "draft": True,
    "stream": True,
    "stream_options": {"include_usage": True},
    "chat_template_kwargs": {"enable_thinking": False},
}
print(json.dumps(body, ensure_ascii=False))

"""
convert_csv.py
--------------
Converts conversations.csv to messages.json required by the RAG pipeline.

CSV format expected: one conversation per row, each row contains lines like:
    User 1: Hello!
    User 2: Hi there!

Output: ../data/messages.json
"""

import csv, io, json, re
from pathlib import Path

CSV_FILE     = Path(__file__).parent / "conversations.csv"
DATA_DIR     = Path(__file__).parent.parent / "data"
MESSAGES_OUT = DATA_DIR / "messages.json"


def convert():
    DATA_DIR.mkdir(exist_ok=True)

    with open(CSV_FILE, encoding="utf-8", errors="replace", newline="") as f:
        content = f.read()

    reader = csv.reader(io.StringIO(content))
    rows   = list(reader)

    messages   = []
    global_id  = 0

    for conv_id, row in enumerate(rows, start=1):
        if not row:
            continue
        conv_text = row[0]
        for line in conv_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(User \d+):\s*(.*)", line)
            if m:
                speaker = m.group(1)
                text    = m.group(2).strip()
                if text:
                    messages.append({
                        "global_id": global_id,
                        "conv_id":   conv_id,
                        "speaker":   speaker,
                        "text":      text,
                    })
                    global_id += 1

    with open(MESSAGES_OUT, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

    print(f"✅ Converted {len(rows):,} conversations → {len(messages):,} messages")
    print(f"   Saved to: {MESSAGES_OUT}")


if __name__ == "__main__":
    convert()

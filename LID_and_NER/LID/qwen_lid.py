import os
import re
import time
import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from openai import (
    OpenAI,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)

with open("prompts/lid_prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

DATASET = "dataset_path"
df = pd.read_csv(DATASET)

load_dotenv()
api_key = os.getenv("API_KEY")

if api_key is None:
    raise RuntimeError("API key not found in environment variables.")

print("API key loaded successfully.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

checkpoint_file = output_dir / "qwen_lid_checkpoint.csv"

#Checkpoint file

if checkpoint_file.exists():
    checkpoint_df = pd.read_csv(checkpoint_file)

    if len(checkpoint_df) == len(df):
        df = checkpoint_df
        print("Checkpoint loaded.")
    else:
        print("Checkpoint size mismatch. Starting over.")

if "qwen_langid" not in df.columns:
    df["qwen_langid"] = pd.NA


# *** LID: Qwen *** #
def get_lid_labels(post, retries=6):
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="qwen/qwen3-8b",
                temperature=0,
                max_tokens=2048,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": post,
                    },
                ],
            )
            return completion.choices[0].message.content

        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        ) as e:
            wait = min(2 ** attempt, 60)
            print(
                f"\nRequest failed ({type(e).__name__}). "
                f"Retrying in {wait} seconds..."
            )
            time.sleep(wait)
    raise RuntimeError("Maximum retries exceeded.")


def parse_lid_output(result):
    labels = {}
    pattern = re.compile(
        r"^\s*(\d+)\s*\|\s*(TR|EN|MIXED|OTHER|NE|AMBIGUOUS)\s*$"
    )
    for line in result.splitlines():
        match = pattern.match(line.strip())
        if match:
            idx = int(match.group(1))
            tag = match.group(2)
            labels[idx] = tag
    return labels

# *** COLLECTING LABELS *** #
alignment_rows = []
token_debug_rows = []

processed_docs = 0
doc_groups = list(df.groupby("doc_id"))

for doc_id, group in tqdm(doc_groups, desc="Processing posts"):

    # Skip documents already completed
    if group["qwen_langid"].notna().all() and not (group["qwen_langid"] == "UNK").any():
        continue
    numbered_tokens = "\n".join(
        f"{i}\t{token}"
        for i, token in enumerate(group["token"].tolist(), start=1)
    )
    try:
        result = get_lid_labels(numbered_tokens)

    except Exception as e:
        print(f"\nFailed on document {doc_id}")
        print(e)
        continue

    label_dict = parse_lid_output(result)

    expected_indices = list(range(1, len(group) + 1))
    returned_indices = sorted(label_dict.keys())


    missing_indices = sorted(set(expected_indices) - set(returned_indices))
    extra_indices = sorted(set(returned_indices) - set(expected_indices))
    aligned = (returned_indices == expected_indices)

    alignment_rows.append({
        "doc_id": doc_id,
        "n_tokens": len(group),
        "aligned": aligned,
        "n_returned_labels": len(returned_indices),
        "missing_indices": str(missing_indices),
        "extra_indices": str(extra_indices),
        "input": numbered_tokens,
        "raw_output": result,
    })

    ordered_labels = [
        label_dict.get(i, "UNK")
        for i in expected_indices
    ]
    
    df.loc[group.index, "qwen_langid"] = ordered_labels

    for i, (_, row) in enumerate(group.iterrows(), start=1):
        token_debug_rows.append({
            "doc_id": row["doc_id"],
            "sent_id": row["sent_id"],
            "tok_id": row["tok_id"],
            "token_index": i,
            "token": row["token"],
            "gold_label": row["lid"],
            "predicted_label": ordered_labels[i-1]
        })

    processed_docs += 1

    # Save every 25 documents
    if processed_docs % 25 == 0:
        df.to_csv(checkpoint_file, index=False)
        print(f"\nCheckpoint saved ({processed_docs} documents).")

    # Small delay to reduce chance of rate limiting
    time.sleep(0.5)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

final_file = output_dir / f"qwen_lid_{timestamp}.csv"
alignment_file = output_dir / (f"qwen_lid_alignment_{timestamp}.csv")
token_debug_file = output_dir / (f"qwen_lid_token_debug_{timestamp}.csv")

df.to_csv(final_file, index=False)
pd.DataFrame(alignment_rows).to_csv(alignment_file, index=False)
pd.DataFrame(token_debug_rows).to_csv(token_debug_file, index=False)

# Remove checkpoint after successful completion
if checkpoint_file.exists():
    checkpoint_file.unlink()

print(f"\nFinished!\nSaved to:\n{final_file}")
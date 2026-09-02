import os
import pandas as pd
import datetime
import re
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

with open("prompts/lid_prompt_few_shot.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

DATASET = "../data_input/cleaned_annotated_dataset.csv"
df = pd.read_csv(DATASET)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment.")

client = OpenAI(api_key=api_key)


# ============================================================
# LID
# ============================================================

def get_lid_labels(post):
    response = client.responses.create(
        model="gpt-4o",
        instructions=system_prompt,
        input=post,
        temperature=0
    )
    return response.output_text


def parse_lid_output(result):
    """
    Parse model output of the form

    1|TR
    2|EN
    3|NE
    """

    labels = {}

    pattern = re.compile(
        r"^\s*(\d+)\s*\|\s*(TR|EN|MIXED|OTHER|NE|AMBIGUOUS)\s*$"
    )

    for line in result.splitlines():

        line = line.strip()

        match = pattern.match(line)

        if match:
            idx = int(match.group(1))
            tag = match.group(2)
            labels[idx] = tag

    return labels


# ============================================================
# Collect predictions
# ============================================================

prediction_rows = []
total_posts = df["doc_id"].nunique()

for doc_id, group in tqdm(
    df.groupby("doc_id"),
    total=total_posts,
    desc="Processing posts"
):

    numbered_tokens = "\n".join(
        f"{i}\n{token}"
        for i, token in enumerate(group["token"].tolist(), start=1)
    )

    result = get_lid_labels(numbered_tokens)
    label_dict = parse_lid_output(result)

    expected_indices = list(range(1, len(group) + 1))
    returned_indices = sorted(label_dict.keys())

    missing_indices = sorted(
        set(expected_indices) - set(returned_indices)
    )

    extra_indices = sorted(
        set(returned_indices) - set(expected_indices)
    )

    aligned = returned_indices == expected_indices

    if aligned:
        ordered_labels = [
            label_dict[i]
            for i in expected_indices
        ]

    else:
        ordered_labels = ["UNK"] * len(group)

    for i, (_, row) in enumerate(group.iterrows(), start=1):

        prediction_rows.append({
            "doc_id": row["doc_id"],
            "sent_id": row["sent_id"],
            "tok_id": row["tok_id"],
            "token": row["token"],
            "lid": row["lid"],
            "borrowed_suffix": row["borrowed_suffix"],
            "ner": row["ner"],
            "gpt_langid":ordered_labels[i-1]
        })


# ============================================================
# Save predictions
# ============================================================

prediction_df = pd.DataFrame(prediction_rows)
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

token_prediction_file = output_dir / f"gpt_token_lid_few_shot{timestamp}.csv"

prediction_df.to_csv(
    token_prediction_file,
    index=False
)
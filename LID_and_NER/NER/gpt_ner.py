import os
import pandas as pd
import datetime
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import re
from dotenv import load_dotenv


load_dotenv()

with open("prompts/ner_prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()


DATASET = "../data_input/cleaned_annotated_dataset.csv"
df = pd.read_csv(DATASET)


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment.")

client = OpenAI(api_key=api_key)



# ============================================================
# NER
# ============================================================

def get_ner_labels(post):

    response = client.responses.create(
        model="gpt-4o",
        instructions=system_prompt,
        input=post,
        temperature=0
    )

    return response.output_text



def parse_ner_output(result):

    labels = {}

    pattern = re.compile(
        r"^\s*(\d+)\s*(?:\||\t)\s*"
        r"(B-PER|I-PER|B-ORG|I-ORG|"
        r"B-LOC|I-LOC|"
        r"B-GROUP|I-GROUP|"
        r"B-PROD|I-PROD|"
        r"B-TITLE|I-TITLE|"
        r"B-EVENT|I-EVENT|"
        r"B-TIME|I-TIME|"
        r"B-OTHER|I-OTHER|O)"
        r"\s*$"
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

alignment_rows = []

total_posts = df["doc_id"].nunique()


for doc_id, group in tqdm(
    df.groupby("doc_id"),
    total=total_posts,
    desc="Processing posts"
):

    numbered_tokens = "\n".join(
        f"{i}\t{token}"
        for i, token in enumerate(
            group["token"].tolist(),
            start=1
        )
    )


    result = get_ner_labels(numbered_tokens)

    label_dict = parse_ner_output(result)



    expected_indices = list(
        range(1, len(group) + 1)
    )

    returned_indices = sorted(
        label_dict.keys()
    )


    missing_indices = sorted(
        set(expected_indices) -
        set(returned_indices)
    )

    extra_indices = sorted(
        set(returned_indices) -
        set(expected_indices)
    )


    aligned = (
        returned_indices ==
        expected_indices
    )



    # -------------------------------
    # Save debugging information
    # -------------------------------

    alignment_rows.append({

        "doc_id": doc_id,

        "n_tokens":
            len(group),

        "aligned":
            aligned,

        "n_returned_labels":
            len(label_dict),

        "expected_indices":
            str(expected_indices),

        "returned_indices":
            str(returned_indices),

        "missing_indices":
            str(missing_indices),

        "extra_indices":
            str(extra_indices),

        "input":
            numbered_tokens,

        "raw_output":
            result,

        "output_length":
            len(result)

    })



    # -------------------------------
    # Create ordered labels
    # -------------------------------

    ordered_labels = [
        label_dict.get(i, "UNK")
        for i in expected_indices
    ]



    # -------------------------------
    # Save token predictions
    # -------------------------------

    for i, (_, row) in enumerate(
        group.iterrows(),
        start=1
    ):

        prediction_rows.append({

            "doc_id":
                row["doc_id"],

            "sent_id":
                row["sent_id"],

            "tok_id":
                row["tok_id"],

            "token":
                row["token"],

            "gold_ner":
                row["ner"],

            "gpt_ner":
                ordered_labels[i-1]

        })



# ============================================================
# Save outputs
# ============================================================

output_dir = Path("output")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)


timestamp = datetime.datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
)


prediction_file = (
    output_dir /
    f"gpt_token_ner_{timestamp}.csv"
)


debug_file = (
    output_dir /
    f"gpt_ner_alignment_debug_{timestamp}.csv"
)



pd.DataFrame(
    prediction_rows
).to_csv(
    prediction_file,
    index=False
)


pd.DataFrame(
    alignment_rows
).to_csv(
    debug_file,
    index=False
)



print("\nFinished")
print("-------------------------")
print("Predictions:")
print(prediction_file)

print("\nDebug:")
print(debug_file)
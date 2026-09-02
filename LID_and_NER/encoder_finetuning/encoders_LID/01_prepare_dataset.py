import os
import json
import random
import pandas as pd

from datasets import Dataset, DatasetDict

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "../../data_input/cleaned_annotated_dataset.csv"      # Change if using .xlsx
OUTPUT_DIR = "processed_data"

RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

LID_LABELS = [
    "TR",
    "EN",
    "MIXED",
    "OTHER",
    "NE",
    "AMBIGUOUS"
]

label2id = {label: i for i, label in enumerate(LID_LABELS)}
id2label = {i: label for label, i in label2id.items()}


# ============================================================
# Load data
# ============================================================

print("Loading annotations...")

df = pd.read_csv(INPUT_FILE)


required_columns = [
    "doc_id",
    "sent_id",
    "token",
    "lid"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")


print("Grouping tokens into sentences...")

documents = {}

for _, row in df.iterrows():

    doc = row["doc_id"]          # string like "post_0001_1", not numeric
    sent = int(row["sent_id"])

    if doc not in documents:
        documents[doc] = {}

    if sent not in documents[doc]:
        documents[doc][sent] = {
            "tokens": [],
            "labels": []
        }

    documents[doc][sent]["tokens"].append(str(row["token"]))
    documents[doc][sent]["labels"].append(label2id[row["lid"]])

print(f"Documents: {len(documents)}")

# ============================================================
# Split by document
# ============================================================

doc_ids = list(documents.keys())

random.seed(RANDOM_SEED)
random.shuffle(doc_ids)

n_docs = len(doc_ids)

n_train = int(TRAIN_RATIO * n_docs)
n_val = int(VAL_RATIO * n_docs)

train_docs = doc_ids[:n_train]
val_docs = doc_ids[n_train:n_train+n_val]
test_docs = doc_ids[n_train+n_val:]

print(f"Train documents: {len(train_docs)}")
print(f"Validation documents: {len(val_docs)}")
print(f"Test documents: {len(test_docs)}")


# ============================================================
# Convert documents to sentence examples
# ============================================================

def build_examples(doc_list):

    examples = []

    for doc in doc_list:

        for sent_id in sorted(documents[doc].keys()):

            sentence = documents[doc][sent_id]

            examples.append({
                "document_id": doc,
                "sentence_id": sent_id,
                "tokens": sentence["tokens"],
                "labels": sentence["labels"]
            })

    return examples


train_examples = build_examples(train_docs)
val_examples = build_examples(val_docs)
test_examples = build_examples(test_docs)

print()
print("Sentence counts")
print("----------------")
print("Train:", len(train_examples))
print("Validation:", len(val_examples))
print("Test:", len(test_examples))

# ============================================================
# Build Hugging Face Dataset
# ============================================================

dataset = DatasetDict({

    "train": Dataset.from_list(train_examples),

    "validation": Dataset.from_list(val_examples),

    "test": Dataset.from_list(test_examples)

})

# ============================================================
# Save
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset.save_to_disk(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "label2id.json"), "w") as f:
    json.dump(label2id, f, indent=4)

with open(os.path.join(OUTPUT_DIR, "id2label.json"), "w") as f:
    json.dump(id2label, f, indent=4)

split_info = {
    "train_documents": train_docs,
    "validation_documents": val_docs,
    "test_documents": test_docs,
    "seed": RANDOM_SEED
}

with open(os.path.join(OUTPUT_DIR, "split_info.json"), "w") as f:
    json.dump(split_info, f, indent=4)

print()
print("Done.")
print(dataset)
#!/usr/bin/env python3
import re, sys
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
HANDLE_RE = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#[^\s]+")
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]+", re.UNICODE)

def tokenize_text(text:str):
    def protect(m):
        return f" {m.group(0)} "
    for rx in [URL_RE, HANDLE_RE, HASHTAG_RE, EMOJI_RE]:
        text = rx.sub(protect, text)
    
    raw = [t for t in re.split(r"\s+", text.strip()) if t]
    tokens = []
    for t in raw:
        if URL_RE.fullmatch(t) or HANDLE_RE.fullmatch(t) or HASHTAG_RE.fullmatch(t) or EMOJI_RE.fullmatch(t):
            tokens.append(t)
        else:
            m = re.match(r"^(.*?)([.!?]+)$", t)
            #Turkish suffix heuristic
            if m and not ("'" in m.group(1) and m.group(1).endswith("e")): 
                core, punc = m.group(1), m.group(2)
                if core: tokens.append(core)
                tokens.append(punc)
            else:
                tokens.append(t)
    return tokens

def process_row(row):
    post_id = f"post_{row.name + 1:03d}" 
    tokens = tokenize_text(str(row['entry']))
    
    token_data = []
    for i, t in enumerate(tokens):
        token_data.append({
            "doc_id": post_id,
            "sent_id": "1",
            "tok_id": i,
            "token": t,
            "lid": "",
            "borrowed_suffix": "", 
            "ner": "O"
        })
    return token_data

if __name__ == "__main__":
    input_csv = "code_mixed_posts_for_tokenizing.csv"
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Could not find {input_csv}")
        sys.exit(1)

    #process and explode
    print(f"Tokenizing {len(df)} posts...")
    token_list = df.apply(process_row, axis=1).explode().tolist()
    
    #final dataframe
    columns = ["doc_id", "sent_id", "tok_id", "token", "lid", "borrowed_suffix", "ner"]
    final_df = pd.DataFrame(token_list)[columns]
    
    output_file = "posts_for_annotation.tsv"
    final_df.to_csv(output_file, sep="\t", index=False, encoding='utf-8')
    print(f"Done! Created {output_file} with {len(final_df)} tokens.")
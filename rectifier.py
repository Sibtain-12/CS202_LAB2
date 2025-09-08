import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
import os

IN_DIFFS = "outputs/diff_with_llm.csv"
OUT_RECTIFIED = "outputs/rectified_commits.csv"
MODEL_NAME = "mamiksik/CommitPredictorT5"

df = pd.read_csv(IN_DIFFS)

# group by commit-hash
grouped = df.groupby("Hash")

# create the summarizer LLM (or reuse the same model)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
nlp = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if __import__("torch").cuda.is_available() else -1)

out_rows = []
for commit_hash, group in tqdm(grouped, desc="rectifying commits"):
    orig_msg = group["Message"].iloc[0]
    per_file_msgs = list(group["llm_message"].fillna("").values)
    per_file_types = list(group["llm_fix_type"].fillna("").values)
    filenames = list(group["Filename"].values)

    # Simple aggregation: join file messages with file names
    aggregated = "\n".join([f"{fn}: {m}" for fn, m in zip(filenames, per_file_msgs) if m.strip()])

    # Use LLM to create a short rectified commit message
    prompt = (
        f"Original commit message: {orig_msg}\n\n"
        f"Per-file messages:\n{aggregated}\n\n"
        "Produce a single concise commit message (present tense) that summarizes the changes across files. "
        "Keep it to 10-20 words."
    )
    try:
        rect_msg = nlp(prompt, max_length=64)[0]["generated_text"].strip()
    except Exception as e:
        rect_msg = " ".join(per_file_msgs)[:120]  # fallback

    # Decide rectified fix type (majority vote)
    types = [t for t in per_file_types if isinstance(t, str) and t]
    fix_type = max(set(types), key=types.count) if types else ""

    out_rows.append({
        "Hash": commit_hash,
        "Original_Message": orig_msg,
        "Rectified_Message": rect_msg,
        "Per_file_aggregated": aggregated,
        "Rectified_fix_type": fix_type,
        "Num_files": len(group)
    })

pd.DataFrame(out_rows).to_csv(OUT_RECTIFIED, index=False)
print("Wrote", OUT_RECTIFIED)

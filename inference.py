# inference.py
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
import os

IN_CSV = "outputs/diff_extraction.csv"
OUT_CSV = "outputs/diff_with_llm.csv"
MODEL_NAME = "mamiksik/CommitPredictorT5"

# Load CSV
df = pd.read_csv(IN_CSV)

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
nlp = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if __import__("torch").cuda.is_available() else -1
)

# Helper function to safely cast and truncate
def safe_str(x, limit=2000):
    if pd.isna(x):
        return ""
    return str(x)[:limit]

def make_prompt(row):
    diff = safe_str(row.get("Diff"))
    before = safe_str(row.get("Source_Code_before"))
    after = safe_str(row.get("Source_Code_after"))
    filename = safe_str(row.get("Filename"))
    orig_msg = safe_str(row.get("Message"))

    prompt = (
        f"File: {filename}\n"
        f"Original commit message: {orig_msg}\n"
        f"Diff:\n{diff}\n\n"
        f"Source before (truncated):\n{before}\n\n"
        f"Source after (truncated):\n{after}\n\n"
        "Write a concise, precise commit message (present-tense) describing the fix for this file. "
        "Also give a short fix type (bugfix/refactor/test/doc/other). "
        "Answer in the format: <MESSAGE> ||| <FIX_TYPE>"
    )
    return prompt

llm_msgs = []
llm_types = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM inference"):
    prompt = make_prompt(row)
    try:
        out = nlp(prompt, max_length=64, truncation=True)[0]["generated_text"].strip()
    except Exception as e:
        out = ""
    if "|||" in out:
        msg, ftype = [s.strip() for s in out.split("|||", 1)]
    else:
        # fallback: guess fix type
        parts = out.rsplit(" ", 1)
        msg = parts[0].strip() if parts else out
        ftype = parts[1].strip() if len(parts) > 1 else "other"
    llm_msgs.append(msg)
    llm_types.append(ftype)

df["llm_message"] = llm_msgs
df["llm_fix_type"] = llm_types
df.to_csv(OUT_CSV, index=False)
print("Wrote", OUT_CSV)

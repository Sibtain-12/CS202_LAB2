# mine.py
import re
import os
import pandas as pd
from pydriller import Repository
from tqdm import tqdm

# CONFIG
REPO = "https://github.com/python-websockets/websockets"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Keyword regex for bug-fix commits
KEYWORD_RE = re.compile(
    r"\b(fix|fixes|fixed|bug|crash|resolv(?:e|es|ed)|close|closes|closed|issue|regression|assertion|testcase|failure|error|hang|leak|overflow|workaround|avoid)\b",
    re.I
)

commits = []
diffs = []

print("Traversing commits...")
for commit in tqdm(Repository(REPO).traverse_commits(), desc="Commits"):
    msg = commit.msg or ""
    if KEYWORD_RE.search(msg):
        # Commit-level info
        commits.append({
            "Hash": commit.hash,
            "Author": commit.author.name if commit.author else None,
            "Date": commit.author_date.isoformat() if commit.author_date else None,
            "Message": msg,
            "Hashes_of_parents": [h for h in commit.parents],
            "Is_merge_commit": len(commit.parents) > 1,
            "List_of_modified_files": [m.filename for m in commit.modified_files]
        })

        # File-level info
        for m in commit.modified_files:
            diffs.append({
                "Hash": commit.hash,
                "Message": msg,
                "Filename": m.filename,
                "Source_Code_before": m.source_code_before if m.source_code_before else "",
                "Source_Code_after": m.source_code if m.source_code else "",
                "Diff": m.diff if m.diff else ""
            })

# Save CSVs
df_commits = pd.DataFrame(commits)
df_diffs = pd.DataFrame(diffs)

df_commits.to_csv(os.path.join(OUT_DIR, "bug_fixing_commits.csv"), index=False)
df_diffs.to_csv(os.path.join(OUT_DIR, "diff_extraction.csv"), index=False)

print("Done! Files saved in", OUT_DIR)



## part e


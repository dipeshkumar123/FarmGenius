import os
import re

md_path = r"d:\Projects\FarmGenius\FLUTTER_CODE.md"
with open(md_path, "r", encoding="utf-8") as f:
    content = f.read()

# Pattern matches:
# ### `app/filepath`
# ```lang
# code
# ```
pattern = r"### `(app/[^`]+)`\s*```[a-z]*\n(.*?)```"

matches = re.finditer(pattern, content, re.DOTALL)
for match in matches:
    filepath = match.group(1)
    file_content = match.group(2)
    
    # Ensure directory exists
    full_path = os.path.join(r"d:\Projects\FarmGenius", filepath)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(file_content)

# Also extract the github actions
pattern2 = r"### `(.github/[^`]+)`\s*```[a-z]*\n(.*?)```"
matches2 = re.finditer(pattern2, content, re.DOTALL)
for match in matches2:
    filepath = match.group(1)
    file_content = match.group(2)
    full_path = os.path.join(r"d:\Projects\FarmGenius", filepath)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(file_content)

print("Flutter Extraction complete.")

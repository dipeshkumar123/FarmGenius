import os
import re

md_path = r"d:\Projects\FarmGenius\BACKEND_CODE.md"
with open(md_path, "r", encoding="utf-8") as f:
    content = f.read()

# Pattern matches:
# ### `backend/filepath`
# ```lang
# code
# ```
pattern = r"### `(backend/[^`]+)`\s*```[a-z]*\n(.*?)```"

matches = re.finditer(pattern, content, re.DOTALL)
for match in matches:
    filepath = match.group(1)
    file_content = match.group(2)
    
    # Ensure directory exists
    full_path = os.path.join(r"d:\Projects\FarmGenius", filepath)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(file_content)
        
print("Extraction complete.")

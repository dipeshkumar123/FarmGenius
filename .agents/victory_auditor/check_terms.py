import os
import json
import sys

# Import functions from verification script dynamically or just re-implement
def parse_glossary_by_row(glossary_path, lang_name):
    rows_terms = []
    with open(glossary_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header_idx = -1
    col_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('|') and 'English Term' in line:
            header_idx = i
            cols = [c.strip() for c in line.split('|')[1:-1]]
            if lang_name in cols:
                col_idx = cols.index(lang_name)
            break
            
    if header_idx == -1 or col_idx == -1:
        raise ValueError(f"Could not find language column for '{lang_name}' in glossary file.")
        
    for line in lines[header_idx + 2:]:
        line = line.strip()
        if not line.startswith('|'):
            continue
        cols = [c.strip() for c in line.split('|')[1:-1]]
        if len(cols) <= col_idx:
            continue
        cell_val = cols[col_idx]
        if not cell_val or cell_val.startswith('---'):
            continue
            
        parts = cell_val.split('/')
        row_alternatives = []
        for part in parts:
            part = part.strip()
            if '(' in part:
                part = part.split('(')[0].strip()
            if part:
                row_alternatives.append(part)
        if row_alternatives:
            english_term = cols[0]
            rows_terms.append((english_term, row_alternatives))
    return rows_terms

def get_all_string_values(d):
    values = []
    for v in d.values():
        if isinstance(v, dict):
            values.extend(get_all_string_values(v))
        elif isinstance(v, str):
            values.append(v)
    return values

def check_lang(lang_code, lang_name):
    project_root = r"d:\Projects\FarmGenius"
    lang_path = os.path.join(project_root, "frontend", "src", "locales", f"{lang_code}.json")
    glossary_path = os.path.join(project_root, "agricultural_glossary.md")
    
    with open(lang_path, 'r', encoding='utf-8') as f:
        lang_data = json.load(f)
    
    glossary_rows = parse_glossary_by_row(glossary_path, lang_name)
    all_values = get_all_string_values(lang_data)
    
    present = []
    absent = []
    
    for eng, alternatives in glossary_rows:
        found = False
        matched_alt = None
        for alt in alternatives:
            for val in all_values:
                if alt in val:
                    found = True
                    matched_alt = alt
                    break
            if found:
                break
        if found:
            present.append((eng, alternatives, matched_alt))
        else:
            absent.append((eng, alternatives))
            
    print(f"\n=== Report for {lang_name} ({lang_code}) ===")
    print(f"Total glossary terms: {len(glossary_rows)}")
    print(f"Present terms ({len(present)}):")
    for eng, alts, matched in present:
        print(f"  - {eng} -> matched '{matched}' (from alternatives {alts})")
    print(f"Absent terms ({len(absent)}):")
    for eng, alts in absent:
        print(f"  - {eng} -> not found (alternatives {alts})")

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    check_lang('hi', 'Hindi')
    check_lang('te', 'Telugu')
    check_lang('kn', 'Kannada')
    check_lang('ta', 'Tamil')
    check_lang('mr', 'Marathi')

#!/usr/bin/env python3
import json
import sys
import os

def get_all_string_values(d):
    values = []
    for v in d.values():
        if isinstance(v, dict):
            values.extend(get_all_string_values(v))
        elif isinstance(v, str):
            values.append(v)
    return values

def check_structure(en_dict, lang_dict, path=""):
    missing = []
    type_mismatch = []
    for k, v in en_dict.items():
        current_path = f"{path}.{k}" if path else k
        if k not in lang_dict:
            missing.append(current_path)
        else:
            if isinstance(v, dict):
                if not isinstance(lang_dict[k], dict):
                    type_mismatch.append(f"{current_path}: expected dict, got {type(lang_dict[k]).__name__}")
                else:
                    m, t = check_structure(v, lang_dict[k], current_path)
                    missing.extend(m)
                    type_mismatch.extend(t)
    return missing, type_mismatch

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

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # Python versions or environments where reconfigure is not available
        
    if len(sys.argv) < 2:
        print("Usage: python verify_translation.py <lang_code>")
        print("Example: python verify_translation.py hi")
        sys.exit(1)
        
    lang = sys.argv[1].lower()
    lang_map = {
        'hi': 'Hindi',
        'kn': 'Kannada',
        'te': 'Telugu',
        'ta': 'Tamil',
        'mr': 'Marathi'
    }
    
    if lang not in lang_map:
        print(f"Error: Unsupported language code '{lang}'. Supported: hi, kn, te, ta, mr")
        sys.exit(1)
        
    lang_name = lang_map[lang]
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)
    
    en_path = os.path.join(project_root, "frontend", "src", "locales", "en.json")
    lang_path = os.path.join(project_root, "frontend", "src", "locales", f"{lang}.json")
    glossary_path = os.path.join(project_root, "agricultural_glossary.md")
    
    print(f"=== Translation Verification for '{lang}' ({lang_name}) ===")
    
    # 1. Parse en.json
    try:
        with open(en_path, 'r', encoding='utf-8') as f:
            en_data = json.load(f)
    except Exception as e:
        print(f"Error: Failed to parse en.json: {e}")
        sys.exit(1)
        
    # 2. Parse lang.json (valid JSON check)
    try:
        with open(lang_path, 'r', encoding='utf-8') as f:
            lang_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: {lang}.json is not valid JSON: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: {lang}.json file not found at {lang_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read {lang}.json: {e}")
        sys.exit(1)
        
    print(f"[PASS] JSON syntax validation for {lang}.json")
    
    # 3. Recursive key check
    missing_keys, type_mismatches = check_structure(en_data, lang_data)
    
    has_errors = False
    if missing_keys:
        print(f"\n[FAIL] Missing translation keys in {lang}.json:")
        for k in missing_keys:
            print(f"  - {k}")
        has_errors = True
    else:
        print("[PASS] All translation keys present in English are also in target locale")
        
    if type_mismatches:
        print(f"\n[FAIL] Key structure type mismatches in {lang}.json:")
        for tm in type_mismatches:
            print(f"  - {tm}")
        has_errors = True
        
    # 4. Glossary check
    try:
        glossary_rows = parse_glossary_by_row(glossary_path, lang_name)
    except Exception as e:
        print(f"\nError parsing glossary: {e}")
        sys.exit(1)
        
    all_values = get_all_string_values(lang_data)
    
    present_count = 0
    absent_terms = []
    for eng, alternatives in glossary_rows:
        found = False
        for alt in alternatives:
            for val in all_values:
                if alt in val:
                    found = True
                    break
            if found:
                break
        if found:
            present_count += 1
        else:
            absent_terms.append((eng, alternatives))
            
    total_terms = len(glossary_rows)
    if total_terms == 0:
        print("\nError: Glossary contains 0 terms.")
        sys.exit(1)
        
    coverage_pct = (present_count / total_terms) * 100
    print(f"\nGlossary usage coverage: {present_count}/{total_terms} terms found ({coverage_pct:.2f}%)")
    
    if coverage_pct < 80.0:
        print(f"[FAIL] Glossary terms coverage is {coverage_pct:.2f}%, which is below the 80.0% threshold.")
        print("Missing terms:")
        for eng, alternatives in absent_terms:
            print(f"  - {eng}: {', '.join(alternatives)}")
        has_errors = True
    else:
        print(f"[PASS] Glossary terms coverage is {coverage_pct:.2f}% (>= 80%)")
        
    if has_errors:
        print("\n=== VERIFICATION FAILED ===")
        sys.exit(1)
    else:
        print("\n=== ALL VERIFICATION CHECKS PASSED ===")
        sys.exit(0)

if __name__ == "__main__":
    main()

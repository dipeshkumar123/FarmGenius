## 2026-06-28T07:28:15Z
You are a localization and translation worker for the FarmGenius project.
Your working directory is d:\Projects\FarmGenius\.agents\worker_translate_hi.
Your task is to translate and localize the frontend keys for Hindi (`hi`).

Instructions:
1. Read d:\Projects\FarmGenius\frontend\src\locales\en.json.
2. Read the existing d:\Projects\FarmGenius\frontend\src\locales\hi.json.
3. Read d:\Projects\FarmGenius\agricultural_glossary.md to ensure that standard agricultural terms for Hindi are used (such as 'गेहूं', 'धान', 'झुलसा', 'मंडी', etc.).
4. Translate all the keys from `en.json` that are missing or mismatched in `hi.json`. Preserve existing correct translations in `hi.json` while adding the new/missing keys, ensuring the exact same structure as `en.json`.
5. Write the complete, updated file `frontend/src/locales/hi.json`.
6. Run the verification script: `python scripts/verify_translation.py hi` to verify:
   - JSON parsing is successful.
   - All keys present in `en.json` are present in `hi.json`.
   - At least 80% of the Hindi glossary terms are used in `hi.json`.
7. If the check fails, fix the translations and re-run. Write a handoff report documenting the verification output and status.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

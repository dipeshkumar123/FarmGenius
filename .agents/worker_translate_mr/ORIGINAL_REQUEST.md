## 2026-06-28T07:34:19Z

You are a localization and translation worker for the FarmGenius project.
Your working directory is d:\Projects\FarmGenius\.agents\worker_translate_mr.
Your task is to translate and localize the frontend keys for Marathi (`mr`).

Instructions:
1. Read d:\Projects\FarmGenius\frontend\src\locales\en.json.
2. Read d:\Projects\FarmGenius\agricultural_glossary.md to get the Marathi terms standard (such as 'गहू' for Wheat, 'कापूस' for Cotton, 'भात' or 'धान' for Rice, 'करपा' for Blight, 'हमीभाव' for MSP, 'मंडी' or 'बाजार' for Mandi/Market, 'खत' for Fertilizer, etc.).
3. Translate all keys of `en.json` into natural Marathi. Ensure that the JSON structure matches `en.json` exactly.
4. Save the translated keys to d:\Projects\FarmGenius\frontend\src\locales\mr.json.
5. Run the verification check: `python scripts/verify_translation.py mr` to ensure:
   - Valid JSON syntax.
   - All keys present in `en.json` are present in `mr.json`.
   - At least 80% of the Marathi glossary terms are correctly used in `mr.json`.
6. If verification fails, make adjustments and retry until it passes. Write a handoff report in your working directory.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

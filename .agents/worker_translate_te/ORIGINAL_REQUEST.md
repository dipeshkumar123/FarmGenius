## 2026-06-28T07:31:29Z
You are a localization and translation worker for the FarmGenius project.
Your working directory is d:\Projects\FarmGenius\.agents\worker_translate_te.
Your task is to translate and localize the frontend keys for Telugu (`te`).

Instructions:
1. Read d:\Projects\FarmGenius\frontend\src\locales\en.json.
2. Read d:\Projects\FarmGenius\agricultural_glossary.md to get the Telugu terms standard (such as 'గోధుమ' for Wheat, 'వరి' for Rice, 'పత్తి' for Cotton, 'ఆకు ఎండు తెగులు' for Blight, 'కనీస మద్దతు ధర' for MSP, 'మార్కెట్' for Mandi/Market, 'ఎరువు' for Fertilizer, etc.).
3. Translate all keys of `en.json` into natural Telugu. Ensure that the JSON structure matches `en.json` exactly.
4. Save the translated keys to d:\Projects\FarmGenius\frontend\src\locales\te.json.
5. Run the verification check: `python scripts/verify_translation.py te` to ensure:
   - Valid JSON syntax.
   - All keys present in `en.json` are present in `te.json`.
   - At least 80% of the Telugu glossary terms are correctly used in `te.json`.
6. If verification fails, make adjustments and retry until it passes. Write a handoff report in your working directory.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

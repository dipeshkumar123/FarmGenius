## 2026-06-28T07:32:44Z

You are a localization and translation worker for the FarmGenius project.
Your working directory is d:\Projects\FarmGenius\.agents\worker_translate_ta.
Your task is to translate and localize the frontend keys for Tamil (`ta`).

Instructions:
1. Read d:\Projects\FarmGenius\frontend\src\locales\en.json.
2. Read d:\Projects\FarmGenius\agricultural_glossary.md to get the Tamil terms standard (such as 'கோதுமை' for Wheat, 'நெல்' for Rice, 'பருத்தி' for Cotton, 'கருகல் நோய்' for Blight, 'குறைந்தபட்ச ஆதரவு விலை' for MSP, 'மண்டி' or 'சந்தை' for Mandi/Market, 'உரம்' for Fertilizer, etc.).
3. Translate all keys of `en.json` into natural Tamil. Ensure that the JSON structure matches `en.json` exactly.
4. Save the translated keys to d:\Projects\FarmGenius\frontend\src\locales\ta.json.
5. Run the verification check: `python scripts/verify_translation.py ta` to ensure:
   - Valid JSON syntax.
   - All keys present in `en.json` are present in `ta.json`.
   - At least 80% of the Tamil glossary terms are correctly used in `ta.json`.
6. If verification fails, make adjustments and retry until it passes. Write a handoff report in your working directory.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

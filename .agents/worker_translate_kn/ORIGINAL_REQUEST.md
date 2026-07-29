## 2026-06-28T07:29:51Z

You are a localization and translation worker for the FarmGenius project.
Your working directory is d:\Projects\FarmGenius\.agents\worker_translate_kn.
Your task is to translate and localize the frontend keys for Kannada (`kn`).

Instructions:
1. Read d:\Projects\FarmGenius\frontend\src\locales\en.json.
2. Read d:\Projects\FarmGenius\agricultural_glossary.md to get the Kannada terms standard (such as 'ಗೋಧಿ' for Wheat, 'ಭತ್ತ' for Rice, 'ಹತ್ತಿ' for Cotton, 'ಬೆಂಕಿ ರೋಗ' for Blight, 'ಮಾರುಕಟ್ಟೆ' for Mandi/Market, 'ಕನಿಷ್ಠ ಬೆಂಬಲ ಬೆಲೆ' for MSP, etc.).
3. Translate all keys of `en.json` into natural Kannada. Ensure that the JSON structure matches `en.json` exactly.
4. Save the translated keys to d:\Projects\FarmGenius\frontend\src\locales\kn.json.
5. Run the verification check: `python scripts/verify_translation.py kn` to ensure:
   - Valid JSON syntax.
   - All keys present in `en.json` are present in `kn.json`.
   - At least 80% of the Kannada glossary terms are correctly used in `kn.json`.
6. If verification fails, make adjustments and retry until it passes. Write a handoff report in your working directory.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

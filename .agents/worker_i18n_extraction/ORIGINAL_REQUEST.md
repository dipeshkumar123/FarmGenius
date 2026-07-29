## 2026-06-28T07:24:20Z
You are a software localization and frontend engineering worker for the FarmGenius project.
Your working directory is d:\Projects\FarmGenius\.agents\worker_i18n_extraction.
Your goal is to extract hardcoded UI strings into `en.json` and implement a verification script for translation quality.

Detailed Tasks:
1. Read the audit report d:\Projects\FarmGenius\.agents\explorer_m1\handoff.md.
2. In the React frontend, locate the 7 files identified with hardcoded text:
   - `frontend/src/pages/WeatherPage.tsx`
   - `frontend/src/pages/MarketPage.tsx`
   - `frontend/src/pages/SchemesPage.tsx`
   - `frontend/src/components/layout/AppShell.tsx`
   - `frontend/src/components/ui/StaleDataBanner.tsx`
   - `frontend/src/components/ui/EmptyState.tsx`
   - `frontend/src/components/ui/LoadingCard.tsx`
3. Refactor these files to replace hardcoded strings with `t(...)` translation calls:
   - Ensure you import and use `useTranslation` (from `react-i18next`) in these files.
   - For `WeatherPage.tsx`, ensure standard dates use dynamic locales based on the selected language instead of the hardcoded 'en-IN'.
4. Add all the extracted keys and their English values to `frontend/src/locales/en.json`. Use the JSON keys structure proposed in the Appendix of `explorer_m1/handoff.md`.
5. Create a Python script at `scripts/verify_translation.py` that will be used to automatically verify translation quality. The script must:
   - Accept a language code (e.g. `hi`, `kn`, `te`, `ta`, `mr`) as a command line argument.
   - Read `frontend/src/locales/en.json` and `frontend/src/locales/<lang>.json`.
   - Validate that `<lang>.json` is valid JSON and parses correctly.
   - Check that all keys present in `en.json` are also present in `<lang>.json` (recursive key check).
   - Parse `d:\Projects\FarmGenius\agricultural_glossary.md` to extract the glossary terms for the specified language. The script must:
     * Locate the column matching the language name (Hindi for `hi`, Kannada for `kn`, Telugu for `te`, Tamil for `ta`, Marathi for `mr`).
     * Extract the native script word from the cell (e.g. from `गेहूं (Gehu)` extract `गेहूं`).
     * Check if these native terms appear as a substring in any of the translated values of `<lang>.json`.
     * Calculate if at least 80% of the glossary terms for that language are present in the translations.
   - Exit with code 0 if all checks pass, and code 1 with descriptive messages if any check fails.
6. Verify your changes. Run `npm run build` or similar build checks in `frontend/` to make sure there are no TypeScript or compilation errors. Run the verification script on the existing `hi.json` to verify that it catches missing keys.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

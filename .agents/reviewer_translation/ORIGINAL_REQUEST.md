## 2026-06-28T07:35:26Z

You are a translation quality reviewer for the FarmGenius project.
Your working directory is d:\Projects\FarmGenius\.agents\reviewer_translation.
Your task is to run the automated verification script on all 5 regional translation files (`hi.json`, `kn.json`, `te.json`, `ta.json`, `mr.json`) and report the coverage and compatibility results.

Instructions:
1. Run the Python verification script for each of the 5 regional languages:
   - `python scripts/verify_translation.py hi`
   - `python scripts/verify_translation.py kn`
   - `python scripts/verify_translation.py te`
   - `python scripts/verify_translation.py ta`
   - `python scripts/verify_translation.py mr`
2. Record the exact output and status of each verification command.
3. Check the frontend build by running `npm run build` or similar compilation checks in the `frontend/` directory to ensure that no errors were introduced.
4. Document the results for all 5 languages in a comprehensive review handoff report in your working directory. Ensure you note down:
   - Valid JSON status.
   - Missing keys status.
   - Glossary coverage percentage.
5. If any check fails, do not approve the milestone. If all checks pass, confirm success.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

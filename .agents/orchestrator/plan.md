# Implementation Plan: FarmGenius UI Localization and Translation Quality

## 1. Goals and Requirements
The goal is to update the FarmGenius frontend application to ensure complete UI localization, particularly for Mandi, Weather, and Government Schemes pages, across five regional languages: Hindi (hi), Kannada (kn), Telugu (te), Tamil (ta), and Marathi (mr).

Specifically:
- **R1. Agricultural Glossary**: Research and construct `agricultural_glossary.md` detailing regional terms (crop names, diseases, weather patterns, market terms) for Hindi, Kannada, Telugu, Tamil, and Marathi.
- **R2. Direct JSON Translation (Chunked)**: Process the translation work **one language at a time**. For each language:
  1. Extract the UI keys for that language from `en.json`.
  2. Generate translations using the glossary as a guide.
  3. Write the resulting `<lang>.json` file.
  4. Validate JSON syntax before moving to the next language.
- **R3. Verification After Each Chunk**: After completing each language file, run an automated check that:
  - All keys are present.
  - The file is valid JSON.
  - At least 80% of the glossary terms appear in the translation (simple string match).

---

## 2. Milestone Steps & Tasks

### Milestone 1: Exploration and Localization Audit
- **Goal**: Audit the frontend codebase to locate hardcoded English text.
- **Status**: Completed. Explorer handoff received.

### Milestone 2: Research and Compile Agricultural Glossary
- **Goal**: Research and write a high-quality glossary (`agricultural_glossary.md` in the project root) containing regional terms for the 5 target languages using `FARMER_CORPUS.md` and standard farming terminology.
- **Verification**: `agricultural_glossary.md` exists and contains at least 30+ regional terms (crop names, diseases, weather patterns, mandi/market terms) mapped to English for all 5 target languages.

### Milestones 3-7: Language Translation and Verification (Hindi, Kannada, Telugu, Tamil, Marathi)
- **Goal**: Translate `en.json` to `<lang>.json` using the glossary, then run verification.
- **Steps per language**:
  1. Translate the keys.
  2. Write `<lang>.json`.
  3. Run script to verify that:
     - All keys match `en.json`.
     - Syntactically valid JSON.
     - At least 80% of the glossary terms for this language appear in the translation file.
- **Verification**: The check passes with a clean result. If it fails, report and retry.

---

## 3. Acceptance Criteria Checklist
- [ ] A glossary file (`agricultural_glossary.md`) exists at the root of the project with region-specific farming terminology for Hindi, Kannada, Telugu, Tamil, and Marathi.
- [ ] Each language JSON file (`hi.json`, `kn.json`, `te.json`, `ta.json`, `mr.json`) is fully populated with translations for every key in `en.json`.
- [ ] The glossary terms are demonstrably used within each language file.
- [ ] All 5 regional JSON files are syntactically valid JSON (no missing brackets or unescaped quotes).
- [ ] Automated per‑language verification passes before the next language is processed.

---

## 4. Current Status
- [x] Initialized project briefing, progress files, and implementation plan.
- [x] Milestone 1: Exploration and Localization Audit [DONE]
- [ ] Milestone 2: Agricultural Glossary Construction [Pending]
- [ ] Milestone 3: Hindi Translation [Pending]
- [ ] Milestone 4: Kannada Translation [Pending]
- [ ] Milestone 5: Telugu Translation [Pending]
- [ ] Milestone 6: Tamil Translation [Pending]
- [ ] Milestone 7: Marathi Translation [Pending]


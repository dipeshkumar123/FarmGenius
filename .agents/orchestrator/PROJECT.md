# Project: FarmGenius Translation and Localization

## Architecture
- React frontend with `i18next` for localization.
- Localization files located in `frontend/src/locales/`.
- Glossary standard located at `agricultural_glossary.md` in the project root.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Explore & Audit | Scan pages for hardcoded text and map en.json | None | DONE |
| 2 | Research Glossary | Construct agricultural_glossary.md from FARMER_CORPUS.md | None | DONE |
| 3 | Hindi Translation | Translate en.json to hi.json & verify | M2 | DONE |
| 4 | Kannada Translation | Translate en.json to kn.json & verify | M2 | DONE |
| 5 | Telugu Translation | Translate en.json to te.json & verify | M2 | DONE |
| 6 | Tamil Translation | Translate en.json to ta.json & verify | M2 | DONE |
| 7 | Marathi Translation | Translate en.json to mr.json & verify | M2 | DONE |


## Interface Contracts
- Standard JSON translation file structure: key-value matching `en.json`.
- Glossary mapping structure: English term -> Hindi, Kannada, Telugu, Tamil, Marathi.


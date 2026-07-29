# Original User Request

## Initial Request — 2026-06-27T12:08:09Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview

Update the FarmGenius app to ensure complete UI localization (including Mandi, Weather, and Government Schemes pages) and improve the quality of agricultural translations across all supported regional languages (Hindi, Kannada, Telugu, Tamil, Marathi).

Working directory: d:/Projects/FarmGenius
Integrity mode: development

## Requirements

### R1. High-Quality Agricultural Glossary
Research and compile a regional agricultural glossary containing key terms used by farmers (e.g., crop names, diseases, weather patterns, market terms) for all 5 target languages. This glossary must act as the strict standard for all translations.

### R2. Direct JSON Translation
Ensure all UI text across the app (especially Mandi, Weather, and Government Schemes pages) is extracted into localization keys. Translate the `en.json` base file into the 5 regional JSON files directly using your own capabilities. Do not rely on external API scripts (to avoid rate limits).

## Acceptance Criteria

### Translation Quality & Completeness
- [ ] A glossary file (`agricultural_glossary.md`) exists containing region-specific farming terminology for Hindi, Kannada, Telugu, Tamil, and Marathi.
- [ ] All keys present in `en.json` are fully translated in `hi.json`, `kn.json`, `te.json`, `ta.json`, and `mr.json`.
- [ ] The terms defined in the agricultural glossary are demonstrably used within the JSON translation files.

### Technical Validity
- [ ] All 5 regional JSON files are syntactically valid JSON (no missing brackets or unescaped quotes).

*Next: Wait for the teamwork system to report completion.*

## Follow-up — 2026-06-28T07:20:50Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview

Update the FarmGenius app to ensure complete UI localization (including Mandi, Weather, and Government Schemes pages) and improve the quality of agricultural translations across all supported regional languages (Hindi, Kannada, Telugu, Tamil, Marathi).

Working directory: d:/Projects/FarmGenius
Integrity mode: development

## Requirements

### R1. High-Quality Agricultural Glossary
Research and compile a regional agricultural glossary containing key terms used by farmers (e.g., crop names, diseases, weather patterns, market terms) for all 5 target languages. This glossary must act as the strict standard for all translations.

### R2. Direct JSON Translation (Chunked)
Process the translation work **one language at a time** to keep LLM usage within quota limits. For each language (Hindi, Kannada, Telugu, Tamil, Marathi):
1. Extract the UI keys for that language from `en.json`.
2. Generate translations using the glossary as a guide.
3. Write the resulting `<lang>.json` file.
4. Validate JSON syntax before moving to the next language.

### R3. Verification After Each Chunk
After completing each language file, run an automated check that:
- All keys are present.
- The file is valid JSON.
- At least 80% of the glossary terms appear in the translation (simple string match).
If a check fails, the agents must pause, report the issue, and retry before proceeding.

## Acceptance Criteria

### Translation Quality & Completeness
- [ ] A glossary file (`agricultural_glossary.md`) exists containing region‑specific farming terminology for Hindi, Kannada, Telugu, Tamil, and Marathi.
- [ ] Each language JSON file (`hi.json`, `kn.json`, `te.json`, `ta.json`, `mr.json`) is fully populated with translations for every key in `en.json`.
- [ ] The glossary terms are demonstrably used within each language file.

### Technical Validity
- [ ] All 5 regional JSON files are syntactically valid JSON (no missing brackets or unescaped quotes).
- [ ] Automated per‑language verification passes before the next language is processed.

---
*Next: Wait for the teamwork system to report completion.*

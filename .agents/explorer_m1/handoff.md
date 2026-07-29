# Handoff Report: Frontend Translation & i18n Audit

This report documents the translation audit of the frontend pages (`MarketPage`, `WeatherPage`, `SchemesPage`) and related layout/UI components in the FarmGenius project, identifying all hardcoded English UI text and missing translation keys in `frontend/src/locales/en.json`.

---

## 1. Observation

### Audited Files and Path Reference
- **Pages**:
  1. `frontend/src/pages/MarketPage.tsx`
  2. `frontend/src/pages/WeatherPage.tsx`
  3. `frontend/src/pages/SchemesPage.tsx`
- **Components & Layout**:
  4. `frontend/src/components/layout/AppShell.tsx`
  5. `frontend/src/components/ui/StaleDataBanner.tsx`
  6. `frontend/src/components/ui/EmptyState.tsx`
  7. `frontend/src/components/ui/LoadingCard.tsx`
- **Locales**:
  8. `frontend/src/locales/en.json`
  9. `frontend/src/locales/hi.json`

---

### Exact Hardcoded Text Identified

#### A. `frontend/src/pages/WeatherPage.tsx`
This file **does not import `useTranslation`** or use translation keys at all. All user-facing text is hardcoded:
- **Alert Status Banner (Lines 129-131)**:
  ```tsx
  <span className="font-noto text-amber-800">
    Using estimated forecast — live weather unavailable.
  </span>
  ```
- **Live / Estimate Status Badges (Line 144)**:
  ```tsx
  {error ? '📡 Estimated' : '🟢 Live'}
  ```
- **Weather Condition Labels (Line 154)**:
  ```tsx
  {today.rainfall_mm > 20 ? 'Heavy Rain' : today.rainfall_mm > 5 ? 'Partly Cloudy' : today.max_temp > 35 ? 'Sunny & Hot' : 'Partly Cloudy'}
  ```
- **Meteorological Parameter Labels (Lines 164, 168, 173)**:
  - `"High/Low"`
  - `"Rain"`
  - `"Humid"`
- **Meteorological Units (Line 177, 212)**:
  - `"km/h"` (in `{Math.round(today.wind_kmh)} km/h`)
  - `"mm 🌧️"` (in `{day.rainfall_mm}mm 🌧️`)
- **Fallback Advisories (Lines 62-68)**:
  - `"Conditions look favorable for standard farming activities."`
  - `"Heavy rain expected — avoid spraying pesticides and monitor drainage."`
  - `"High temperatures expected. Ensure adequate crop irrigation."`
- **Section Headings (Lines 190, 221, 255)**:
  - `7-Day Forecast`
  - `Farming Advisories`
  - `Precipitation (mm)`
- **Chart Tooltip Metric (Line 270)**:
  - `'Rainfall'` (in `formatter={((val: any) => [`${Number(val).toFixed(0)} mm`, 'Rainfall']) as any}`)
- **Advisory Default States (Lines 242, 246)**:
  - `"All 7 days: Good conditions for farming activities. Stay updated for any changes."`
- **Day Label (Line 34)**:
  - `"Today"` (in `if (idx === 0) return 'Today';`)

---

#### B. `frontend/src/pages/MarketPage.tsx`
Although this page uses `useTranslation`, several values displayed in the UI are hardcoded:
- **Mock / Database Data Properties (Lines 50-57, 149, 283)**:
  - Time indicators: `"2h ago"`, `"3h ago"`, `"4h ago"`, `"5h ago"`, `"6h ago"`, `"Just now"`.
- **Chart Day Labels (Line 65)**:
  - `"Today"`, `"Mon"`, `"Tue"`, `"Wed"`, `"Thu"`, `"Fri"`, `"Sat"` in `generateTrend`.
- **Market Suffixes & Distances (Lines 72-75)**:
  - Distance unit: `"km"` (in `'12 km'`, `'8 km'`, etc.).
  - APMC suffix: `"APMC"` (in `'Hubli APMC'`, etc.).
- **Crop / Commodity Names (Lines 50-57, 260)**:
  - `"Wheat"`, `"Maize"`, `"Soybean"`, `"Rice"`, `"Tomato"`, `"Onion"`, `"Cotton"`, `"Chickpea"`. These are rendered directly as `{item.name}` without a translating wrapper like `t(item.name)`.

---

#### C. `frontend/src/pages/SchemesPage.tsx`
This page utilizes translations well, except for one hardcoded user-facing string:
- **Chat Prefill Query Template (Lines 148-149)**:
  ```typescript
  prefill: `Am I eligible for ${scheme.name} (${scheme.fullName})? My farm is in ${farmer?.state || 'Karnataka'}.`
  ```

---

#### D. `frontend/src/components/layout/AppShell.tsx`
Contains hardcoded fallback text and accessibility `aria-label` labels:
- **Fallback User Info (Lines 233, 236, 261, 264)**:
  - `'Farmer'` (in `{farmer?.name ?? 'Farmer'}`)
  - `'India'` (in `{farmer?.district ?? 'India'}`)
- **Accessibility `aria-label` attributes (Lines 129, 181, 223, 314, 326, 379)**:
  - `aria-label="Select language"`
  - `aria-label="Notifications"`
  - `aria-label="Profile menu"`
  - `aria-label="FarmGenius home"`
  - `aria-label="Main navigation"`
  - `aria-label="Mobile navigation"`

---

#### E. `frontend/src/components/ui/StaleDataBanner.tsx`
This component **does not import `useTranslation`**. All UI text and accessibility tags are hardcoded:
- **Time/Duration Helper (Lines 44-47)**:
  - `"just now"`, `"minute"`, `"minutes"`, `"ago"`, `"hour"`, `"hours"`.
- **Status Messages & Button Actions (Lines 125-138, 163)**:
  - `"Showing cached data from "`
  - `"Tap to refresh."`
  - `"Updating…"`
  - `"Refresh"`
- **Accessibility `aria-label` attributes (Lines 106, 136, 151, 173, 213)**:
  - `aria-label="Stale data warning"`
  - `aria-label="Refresh data now"`
  - `aria-label={isRefreshing ? 'Refreshing data…' : 'Refresh data'}`
  - `aria-label="Dismiss stale data warning"`

---

#### F. `frontend/src/components/ui/EmptyState.tsx`
This component **does not import `useTranslation`**. It uses hardcoded fallback strings for errors and empty searches:
- **Network Error Preset (Lines 276, 302, 304)**:
  - `"Could not connect to the server. Please check your internet connection and try again."`
  - `"Something went wrong"`
  - `"Try Again"`, `"Trying again…"`
- **No Search Results Preset (Lines 354, 357-358, 360)**:
  - `"No results found"`
  - `"We couldn't find anything for \"${query}\". Try a different crop name or district."`
  - `"No items match your search. Try adjusting your filters."`
  - `"Clear Search"`
- **Generic Loading text (Line 235)**:
  - `"Loading…"`

---

#### G. `frontend/src/components/ui/LoadingCard.tsx`
- **Default labels & Accessibility (Lines 43, 172)**:
  - `aria-label="Loading content"`
  - `label = 'Loading…'`

---

## 2. Logic Chain

1. **Comparison with `en.json`**:
   - `en.json` has standard keys for navigating and viewing basic dashboard tiles, but is completely missing keys for internal weather details, custom empty states, stale banner warnings, accessibility aria-labels, and specific crop/commodity names.
2. **Impact on Multi-lingual Support**:
   - Because `WeatherPage.tsx`, `StaleDataBanner.tsx`, and `EmptyState.tsx` do not import or call `t()`, they will remain strictly in English even if the user switches the language (e.g. to Hindi or Kannada).
   - Because `MarketPage.tsx` renders day labels and crop names from database strings or raw code arrays, these elements cannot be translated dynamically without translation map lookups (e.g., `t(`crops.${item.name}`)`).
3. **Synthesis**:
   - `en.json` (and matching files like `hi.json`) needs to be expanded with structural segments for `weather`, `components` (containing `empty_state`, `stale_banner`, and common `aria_labels`), and `crops`.

---

## 3. Caveats

- **Mock Data Limitations**: Some crop names and APMC locations are mock database records. In production, these will come from the backend database. Translation keys should ideally cover all possible commodities, or the backend should return translated names based on the user's language header.
- **Dynamic Date Labels**: Days of the week in `WeatherPage` are formatted via standard `toLocaleDateString('en-IN', ...)`. While it handles internationalization using the browser's locale, it is hardcoded to `'en-IN'` (line 36). It should dynamically accept the active language locale code (e.g. `hi-IN` or `kn-IN`).

---

## 4. Conclusion

The translation configuration in `en.json` is **incomplete**. Significant portions of the weather detail view, standard state banners, accessibility markers, and commodity lookups are hardcoded in English. 

### Actionable Next Steps
1. Add `useTranslation` to `WeatherPage.tsx`, `StaleDataBanner.tsx`, `EmptyState.tsx`, and `LoadingCard.tsx`.
2. Extract all hardcoded strings into structured translation paths.
3. Append the proposed translation keys (listed below) to `en.json` and matching translation files.

---

## 5. Verification Method

To verify these findings manually:
1. Open `frontend/src/pages/WeatherPage.tsx` and observe the lack of `import { useTranslation }` on lines 1-12.
2. Search `frontend/src/locales/en.json` for keys like `"weather.forecast_title"`, `"components.stale_banner.showing_cached"`, or crop names like `"crops.wheat"`. You will find they do not exist.
3. Switch the app language to Hindi on the dashboard page and navigate to `WeatherPage`. You will observe all content remains in English.

---

## 6. Appendix: Proposed Keys to Add to `en.json`

```json
  "crops": {
    "wheat": "Wheat",
    "maize": "Maize",
    "soybean": "Soybean",
    "rice": "Rice",
    "rice_fine": "Rice (Fine)",
    "tomato": "Tomato",
    "onion": "Onion",
    "cotton": "Cotton",
    "chickpea": "Chickpea"
  },
  "weather": {
    "estimated_status": "Estimated",
    "live_status": "Live",
    "estimated_warning": "Using estimated forecast — live weather unavailable.",
    "forecast_title": "7-Day Forecast",
    "advisories_title": "Farming Advisories",
    "precipitation_title": "Precipitation (mm)",
    "high_low": "High/Low",
    "rain": "Rain",
    "humid": "Humid",
    "conditions": {
      "heavy_rain": "Heavy Rain",
      "partly_cloudy": "Partly Cloudy",
      "sunny_hot": "Sunny & Hot"
    },
    "advisories": {
      "favorable": "Conditions look favorable for standard farming activities.",
      "heavy_rain": "Heavy rain expected — avoid spraying pesticides and monitor drainage.",
      "high_temp": "High temperatures expected. Ensure adequate crop irrigation.",
      "all_clear": "All 7 days: Good conditions for farming activities. Stay updated for any changes."
    },
    "rainfall": "Rainfall",
    "today": "Today"
  },
  "schemes": {
    "chat_prefill": "Am I eligible for {{name}} ({{fullName}})? My farm is in {{state}}."
  },
  "components": {
    "stale_banner": {
      "showing_cached": "Showing cached data from {{time}}.",
      "tap_refresh": "Tap to refresh.",
      "updating": "Updating…",
      "refresh": "Refresh",
      "time": {
        "just_now": "just now",
        "minute": "1 minute ago",
        "minutes": "{{count}} minutes ago",
        "hour": "1 hour ago",
        "hours": "{{count}} hours ago"
      }
    },
    "empty_state": {
      "loading": "Loading…",
      "something_went_wrong": "Something went wrong",
      "try_again": "Try Again",
      "trying_again": "Trying again…",
      "no_results": "No results found",
      "clear_search": "Clear Search",
      "default_network_error": "Could not connect to the server. Please check your internet connection and try again.",
      "no_search_match": "We couldn't find anything for \"{{query}}\". Try a different crop name or district.",
      "no_filters_match": "No items match your search. Try adjusting your filters."
    },
    "loading_card": {
      "loading_content": "Loading content"
    }
  },
  "accessibility": {
    "select_language": "Select language",
    "notifications": "Notifications",
    "profile_menu": "Profile menu",
    "home_link": "FarmGenius home",
    "main_navigation": "Main navigation",
    "mobile_navigation": "Mobile navigation"
  }
```

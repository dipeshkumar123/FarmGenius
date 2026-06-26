# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: security.spec.ts >> TC-SEC-005: XSS payload in chat input is rendered as text, not executed >> script tag in message is not executed
- Location: tests\e2e\security.spec.ts:37:3

# Error details

```
Error: page.goto: net::ERR_CONNECTION_REFUSED at http://localhost:5173/
Call log:
  - navigating to "http://localhost:5173/", waiting until "load"

```

# Test source

```ts
  1  | import type { Page } from '@playwright/test';
  2  | 
  3  | /**
  4  |  * Bypass the UI login flow by injecting a mock JWT into localStorage.
  5  |  * Also writes the Zustand 'farmgenius-app' key so the app thinks the
  6  |  * user is authenticated without hitting the real backend.
  7  |  */
  8  | export async function loginAsDemoFarmer(page: Page): Promise<void> {
  9  |   // Navigate to the app root so we have a same-origin context to set storage
> 10 |   await page.goto('/');
     |              ^ Error: page.goto: net::ERR_CONNECTION_REFUSED at http://localhost:5173/
  11 | 
  12 |   await page.evaluate(() => {
  13 |     // Set the raw JWT that the API client reads from
  14 |     const mockToken =
  15 |       'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.' +
  16 |       'eyJzdWIiOiI5OTk5OTk5OTk5IiwiaWF0IjoxNzAwMDAwMDAwLCJleHAiOjk5OTk5OTk5OTl9.' +
  17 |       'MOCK_SIGNATURE';
  18 |     localStorage.setItem('fg_token', mockToken);
  19 | 
  20 |     // Hydrate Zustand persisted state so ProtectedRoute passes
  21 |     const zustandState = {
  22 |       state: {
  23 |         isAuthenticated: true,
  24 |         language: 'en',
  25 |         isOffline: false,
  26 |         farmer: {
  27 |           name: 'Demo Farmer',
  28 |           phone: '9999999999',
  29 |           district: 'Dharwad',
  30 |           state: 'Karnataka',
  31 |           language: 'en',
  32 |           crops: ['Wheat', 'Tomato', 'Onion'],
  33 |         },
  34 |       },
  35 |       version: 0,
  36 |     };
  37 |     localStorage.setItem('farmgenius-app', JSON.stringify(zustandState));
  38 |   });
  39 | }
  40 | 
```
# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: security.spec.ts >> TC-AUTH-009: All protected routes redirect to /login when unauthenticated >> /dashboard → /login when localStorage is empty
- Location: tests\e2e\security.spec.ts:20:5

# Error details

```
Error: page.goto: net::ERR_CONNECTION_REFUSED at http://localhost:5173/
Call log:
  - navigating to "http://localhost:5173/", waiting until "load"

```

# Test source

```ts
  1  | /**
  2  |  * Security E2E tests — FarmGenius
  3  |  *
  4  |  * Test IDs:
  5  |  *   TC-AUTH-009  All protected routes redirect to /login when unauthenticated
  6  |  *   TC-SEC-005   XSS payload in chat input is rendered as text, not executed
  7  |  */
  8  | 
  9  | import { test, expect } from '@playwright/test';
  10 | import { loginAsDemoFarmer } from './helpers/auth';
  11 | 
  12 | const BACKEND_BASE = 'http://localhost:8001';
  13 | 
  14 | // ─── TC-AUTH-009: Extended protected route check ─────────────────────────────
  15 | 
  16 | test.describe('TC-AUTH-009: All protected routes redirect to /login when unauthenticated', () => {
  17 |   const ALL_PROTECTED = ['/dashboard', '/chat', '/market', '/scan', '/schemes', '/weather'];
  18 | 
  19 |   for (const route of ALL_PROTECTED) {
  20 |     test(`${route} → /login when localStorage is empty`, async ({ page }) => {
  21 |       // Clear all storage before navigating
> 22 |       await page.goto('/');
     |                  ^ Error: page.goto: net::ERR_CONNECTION_REFUSED at http://localhost:5173/
  23 |       await page.evaluate(() => {
  24 |         localStorage.clear();
  25 |         sessionStorage.clear();
  26 |       });
  27 | 
  28 |       await page.goto(route);
  29 |       await expect(page).toHaveURL(/\/login/, { timeout: 10_000 });
  30 |     });
  31 |   }
  32 | });
  33 | 
  34 | // ─── TC-SEC-005: XSS in chat input ───────────────────────────────────────────
  35 | 
  36 | test.describe('TC-SEC-005: XSS payload in chat input is rendered as text, not executed', () => {
  37 |   test('script tag in message is not executed', async ({ page }) => {
  38 |     // Track any unexpected dialog (alert / confirm / prompt)
  39 |     let dialogFired = false;
  40 |     page.on('dialog', async (dialog) => {
  41 |       dialogFired = true;
  42 |       await dialog.dismiss();
  43 |     });
  44 | 
  45 |     // Mock /chat so the XSS payload doesn't need a real backend
  46 |     await page.route(`${BACKEND_BASE}/chat`, (route) => {
  47 |       route.fulfill({
  48 |         status: 200,
  49 |         contentType: 'application/json',
  50 |         body: JSON.stringify({
  51 |           response: 'Here is safe advice for your crop.',
  52 |           source: 'Groq LLM',
  53 |           confidence: 0.9,
  54 |         }),
  55 |       });
  56 |     });
  57 | 
  58 |     // Capture the original page title
  59 |     await loginAsDemoFarmer(page);
  60 |     await page.goto('/chat');
  61 | 
  62 |     const originalTitle = await page.title();
  63 | 
  64 |     // Wait for chat to be ready
  65 |     const chatInput = page.locator('textarea[placeholder*="Ask anything"]');
  66 |     await chatInput.waitFor({ state: 'visible', timeout: 15_000 });
  67 | 
  68 |     // Send an XSS payload as a user message
  69 |     const xssPayload = '<script>document.title="XSS"</script>';
  70 |     await chatInput.fill(xssPayload);
  71 |     await chatInput.press('Enter');
  72 | 
  73 |     // Wait for bot response to arrive (confirms the message cycle completed)
  74 |     await page.locator('.bubble-bot').last().waitFor({ state: 'visible', timeout: 15_000 });
  75 | 
  76 |     // 1. Page title must NOT have been changed to "XSS"
  77 |     const currentTitle = await page.title();
  78 |     expect(currentTitle).not.toBe('XSS');
  79 | 
  80 |     // 2. The title should still match the original (or at minimum not be "XSS")
  81 |     expect(currentTitle).toBe(originalTitle);
  82 | 
  83 |     // 3. No dialog (alert) must have fired
  84 |     expect(dialogFired).toBe(false);
  85 | 
  86 |     // 4. The raw payload text should be visible as text in the user bubble
  87 |     //    (React escapes innerHTML, so it renders as literal characters)
  88 |     const userBubble = page.locator('.bubble-user').last();
  89 |     await expect(userBubble).toContainText('<script>', { timeout: 5_000 });
  90 |   });
  91 | });
  92 | 
```
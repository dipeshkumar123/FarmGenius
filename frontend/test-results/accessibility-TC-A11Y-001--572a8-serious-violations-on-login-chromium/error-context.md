# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: accessibility.spec.ts >> TC-A11Y-001: Login page has no critical axe violations >> axe reports no critical or serious violations on /login
- Location: tests\e2e\accessibility.spec.ts:15:3

# Error details

```
Error: page.goto: net::ERR_CONNECTION_REFUSED at http://localhost:5173/login
Call log:
  - navigating to "http://localhost:5173/login", waiting until "load"

```

# Test source

```ts
  1  | /**
  2  |  * Accessibility E2E tests — FarmGenius
  3  |  *
  4  |  * Test IDs:
  5  |  *   TC-A11Y-001  Login page has no critical axe violations
  6  |  *   TC-A11Y-002  Keyboard-only login flow reaches OTP step
  7  |  */
  8  | 
  9  | import { test, expect } from '@playwright/test';
  10 | import AxeBuilder from '@axe-core/playwright';
  11 | 
  12 | // ─── TC-A11Y-001 ──────────────────────────────────────────────────────────────
  13 | 
  14 | test.describe('TC-A11Y-001: Login page has no critical axe violations', () => {
  15 |   test('axe reports no critical or serious violations on /login', async ({ page }) => {
> 16 |     await page.goto('/login');
     |                ^ Error: page.goto: net::ERR_CONNECTION_REFUSED at http://localhost:5173/login
  17 | 
  18 |     // Wait for the page to be fully rendered before scanning
  19 |     await page.locator('input[type="tel"]').waitFor({ state: 'visible' });
  20 | 
  21 |     const accessibilityScanResults = await new AxeBuilder({ page })
  22 |       // Only flag critical and serious issues — minor / moderate are warnings, not blockers
  23 |       .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
  24 |       .analyze();
  25 | 
  26 |     const criticalOrSerious = accessibilityScanResults.violations.filter(
  27 |       (v) => v.impact === 'critical' || v.impact === 'serious'
  28 |     );
  29 | 
  30 |     // Print summary for debugging when test fails in CI
  31 |     if (criticalOrSerious.length > 0) {
  32 |       console.error(
  33 |         'Axe violations (critical/serious):',
  34 |         criticalOrSerious.map((v) => ({
  35 |           id: v.id,
  36 |           impact: v.impact,
  37 |           description: v.description,
  38 |           nodes: v.nodes.length,
  39 |         }))
  40 |       );
  41 |     }
  42 | 
  43 |     expect(criticalOrSerious).toHaveLength(0);
  44 |   });
  45 | });
  46 | 
  47 | // ─── TC-A11Y-002 ──────────────────────────────────────────────────────────────
  48 | 
  49 | test.describe('TC-A11Y-002: Keyboard-only login flow reaches OTP step', () => {
  50 |   test('tab navigation through phone, terms, send OTP works', async ({ page }) => {
  51 |     await page.goto('/login');
  52 | 
  53 |     // ① Wait for the phone input to be in the DOM and focusable
  54 |     const phoneInput = page.locator('input[type="tel"]');
  55 |     await phoneInput.waitFor({ state: 'visible' });
  56 | 
  57 |     // ② Tab to phone input (first interactive element inside the form area)
  58 |     //    We click the body to reset focus, then Tab into the form
  59 |     await page.locator('body').click();
  60 | 
  61 |     // Focus the phone input directly via Tab (it's among the first tabbable elements)
  62 |     await phoneInput.focus();
  63 |     await phoneInput.fill('9999999999');
  64 | 
  65 |     // ③ Tab to the Terms checkbox toggle div and press Space
  66 |     await page.keyboard.press('Tab');
  67 |     await page.waitForTimeout(300); // Allow focus to settle
  68 |     await page.keyboard.press('Space');
  69 | 
  70 |     // Ensure it actually toggled
  71 |     await expect(page.getByRole('checkbox')).toHaveAttribute('aria-checked', 'true', { timeout: 5000 });
  72 | 
  73 |     // ④ Tab to Send OTP button and press Enter
  74 |     await page.keyboard.press('Tab');
  75 | 
  76 |     // Skip any intermediate tab stops (e.g. ToS link, Privacy link)
  77 |     // We press Tab until the focused element is the Send OTP button
  78 |     let attempts = 0;
  79 |     while (attempts < 6) {
  80 |       const focused = await page.evaluate(() => {
  81 |         const el = document.activeElement;
  82 |         return el?.textContent?.trim() ?? '';
  83 |       });
  84 |       if (/send otp/i.test(focused)) break;
  85 |       await page.keyboard.press('Tab');
  86 |       attempts++;
  87 |     }
  88 | 
  89 |     await page.keyboard.press('Enter');
  90 | 
  91 |     // ⑤ OTP step should now be visible
  92 |     const firstOtpBox = page.locator('input[inputmode="numeric"][maxlength="1"]').first();
  93 |     await expect(firstOtpBox).toBeVisible({ timeout: 10_000 });
  94 | 
  95 |     // ⑥ The heading should confirm we're in OTP step
  96 |     await expect(page.getByText(/verify your number/i)).toBeVisible();
  97 |   });
  98 | });
  99 | 
```
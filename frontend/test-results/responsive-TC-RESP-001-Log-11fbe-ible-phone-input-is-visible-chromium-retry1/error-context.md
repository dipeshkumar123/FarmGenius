# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: responsive.spec.ts >> TC-RESP-001: Login page on mobile viewport (375 × 812) >> left illustration panel is not visible, phone input is visible
- Location: tests\e2e\responsive.spec.ts:15:3

# Error details

```
Error: page.goto: net::ERR_CONNECTION_REFUSED at http://localhost:5173/login
Call log:
  - navigating to "http://localhost:5173/login", waiting until "load"

```

# Test source

```ts
  1  | /**
  2  |  * Responsive E2E tests — FarmGenius
  3  |  *
  4  |  * Test IDs:
  5  |  *   TC-RESP-001  Login page on mobile (375x812) — left illustration panel is hidden
  6  |  */
  7  | 
  8  | import { test, expect } from '@playwright/test';
  9  | 
  10 | // ─── TC-RESP-001 ──────────────────────────────────────────────────────────────
  11 | 
  12 | test.describe('TC-RESP-001: Login page on mobile viewport (375 × 812)', () => {
  13 |   test.use({ viewport: { width: 375, height: 812 } });
  14 | 
  15 |   test('left illustration panel is not visible, phone input is visible', async ({ page }) => {
> 16 |     await page.goto('/login');
     |                ^ Error: page.goto: net::ERR_CONNECTION_REFUSED at http://localhost:5173/login
  17 | 
  18 |     // Wait for the page to render
  19 |     await page.locator('input[type="tel"]').waitFor({ state: 'visible' });
  20 | 
  21 |     // ── Left panel: the LoginPage wraps LeftPanel in a div with class
  22 |     //    "lg:w-[48%] xl:w-[45%] lg:min-h-screen shrink-0"
  23 |     //    At 375px width (below Tailwind's lg breakpoint of 1024px) the
  24 |     //    LeftPanel itself uses "hidden lg:flex" on its inner container.
  25 |     //    We check that the left-panel illustrations are not in view.
  26 |     //
  27 |     //    Strategy: locate the element that contains "AI Crop Advisor in your language"
  28 |     //    which is only inside the LeftPanel component.
  29 |     const leftPanelContent = page.getByText('AI Crop Advisor in your language');
  30 |     await expect(leftPanelContent).not.toBeVisible();
  31 | 
  32 |     // ── The phone input (right-side auth form) MUST be visible on mobile
  33 |     const phoneInput = page.locator('input[type="tel"]');
  34 |     await expect(phoneInput).toBeVisible();
  35 | 
  36 |     // ── The mobile-only logo ("FarmGenius" text with lg:hidden class) should show
  37 |     const mobileLogo = page.locator('.lg\\:hidden').getByText('FarmGenius').first();
  38 |     await expect(mobileLogo).toBeVisible();
  39 | 
  40 |     // ── Send OTP button should be present (may be disabled until phone is valid)
  41 |     const sendOtpBtn = page.getByRole('button', { name: /send otp/i });
  42 |     await expect(sendOtpBtn).toBeVisible();
  43 |   });
  44 | 
  45 |   test('OTP boxes are usable on mobile after valid phone entry', async ({ page }) => {
  46 |     await page.goto('/login');
  47 | 
  48 |     const phoneInput = page.locator('input[type="tel"]');
  49 |     await phoneInput.waitFor({ state: 'visible' });
  50 |     await phoneInput.fill('9999999999');
  51 | 
  52 |     const termsToggle = page.locator('text=I agree to the').locator('..').locator('div').first();
  53 |     await termsToggle.click();
  54 | 
  55 |     await page.getByRole('button', { name: /send otp/i }).click();
  56 | 
  57 |     // OTP boxes should appear and be usable even on a 375px screen
  58 |     const firstOtpBox = page.locator('input[inputmode="numeric"][maxlength="1"]').first();
  59 |     await expect(firstOtpBox).toBeVisible({ timeout: 10_000 });
  60 | 
  61 |     const box = await firstOtpBox.boundingBox();
  62 |     expect(box).not.toBeNull();
  63 |     // Boxes should be reasonably sized on mobile (at least 30px wide)
  64 |     expect(box!.width).toBeGreaterThanOrEqual(30);
  65 |   });
  66 | });
  67 | 
```
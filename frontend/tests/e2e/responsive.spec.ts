/**
 * Responsive E2E tests — FarmGenius
 *
 * Test IDs:
 *   TC-RESP-001  Login page on mobile (375x812) — left illustration panel is hidden
 */

import { test, expect } from '@playwright/test';

// ─── TC-RESP-001 ──────────────────────────────────────────────────────────────

test.describe('TC-RESP-001: Login page on mobile viewport (375 × 812)', () => {
  test.use({ viewport: { width: 375, height: 812 } });

  test('left illustration panel is not visible, phone input is visible', async ({ page }) => {
    await page.goto('/login');

    // Wait for the page to render
    await page.locator('input[type="tel"]').waitFor({ state: 'visible' });

    // ── Left panel: the LoginPage wraps LeftPanel in a div with class
    //    "lg:w-[48%] xl:w-[45%] lg:min-h-screen shrink-0"
    //    At 375px width (below Tailwind's lg breakpoint of 1024px) the
    //    LeftPanel itself uses "hidden lg:flex" on its inner container.
    //    We check that the left-panel illustrations are not in view.
    //
    //    Strategy: locate the element that contains "AI Crop Advisor in your language"
    //    which is only inside the LeftPanel component.
    const leftPanelContent = page.getByText('AI Crop Advisor in your language');
    await expect(leftPanelContent).not.toBeVisible();

    // ── The phone input (right-side auth form) MUST be visible on mobile
    const phoneInput = page.locator('input[type="tel"]');
    await expect(phoneInput).toBeVisible();

    // ── The mobile-only logo ("FarmGenius" text with lg:hidden class) should show
    const mobileLogo = page.locator('.lg\\:hidden').getByText('FarmGenius').first();
    await expect(mobileLogo).toBeVisible();

    // ── Send OTP button should be present (may be disabled until phone is valid)
    const sendOtpBtn = page.getByRole('button', { name: /send otp/i });
    await expect(sendOtpBtn).toBeVisible();
  });

  test('OTP boxes are usable on mobile after valid phone entry', async ({ page }) => {
    await page.goto('/login');

    const phoneInput = page.locator('input[type="tel"]');
    await phoneInput.waitFor({ state: 'visible' });
    await phoneInput.fill('9999999999');

    const termsToggle = page.locator('text=I agree to the').locator('..').locator('div').first();
    await termsToggle.click();

    await page.getByRole('button', { name: /send otp/i }).click();

    // OTP boxes should appear and be usable even on a 375px screen
    const firstOtpBox = page.locator('input[inputmode="numeric"][maxlength="1"]').first();
    await expect(firstOtpBox).toBeVisible({ timeout: 10_000 });

    const box = await firstOtpBox.boundingBox();
    expect(box).not.toBeNull();
    // Boxes should be reasonably sized on mobile (at least 30px wide)
    expect(box!.width).toBeGreaterThanOrEqual(30);
  });
});

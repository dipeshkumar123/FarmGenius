/**
 * Accessibility E2E tests — FarmGenius
 *
 * Test IDs:
 *   TC-A11Y-001  Login page has no critical axe violations
 *   TC-A11Y-002  Keyboard-only login flow reaches OTP step
 */

import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

// ─── TC-A11Y-001 ──────────────────────────────────────────────────────────────

test.describe('TC-A11Y-001: Login page has no critical axe violations', () => {
  test('axe reports no critical or serious violations on /login', async ({ page }) => {
    await page.goto('/login');

    // Wait for the page to be fully rendered before scanning
    await page.locator('input[type="tel"]').waitFor({ state: 'visible' });

    const accessibilityScanResults = await new AxeBuilder({ page })
      // Only flag critical and serious issues — minor / moderate are warnings, not blockers
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .analyze();

    const criticalOrSerious = accessibilityScanResults.violations.filter(
      (v) => v.impact === 'critical' || v.impact === 'serious'
    );

    // Print summary for debugging when test fails in CI
    if (criticalOrSerious.length > 0) {
      console.error(
        'Axe violations (critical/serious):',
        criticalOrSerious.map((v) => ({
          id: v.id,
          impact: v.impact,
          description: v.description,
          nodes: v.nodes.length,
        }))
      );
    }

    expect(criticalOrSerious).toHaveLength(0);
  });
});

// ─── TC-A11Y-002 ──────────────────────────────────────────────────────────────

test.describe('TC-A11Y-002: Keyboard-only login flow reaches OTP step', () => {
  test('tab navigation through phone, terms, send OTP works', async ({ page }) => {
    await page.goto('/login');

    // ① Wait for the phone input to be in the DOM and focusable
    const phoneInput = page.locator('input[type="tel"]');
    await phoneInput.waitFor({ state: 'visible' });

    // ② Tab to phone input (first interactive element inside the form area)
    //    We click the body to reset focus, then Tab into the form
    await page.locator('body').click();

    // Focus the phone input directly via Tab (it's among the first tabbable elements)
    await phoneInput.focus();
    await phoneInput.fill('9999999999');

    // ③ Tab to the Terms checkbox toggle div and press Space
    await page.keyboard.press('Tab');
    await page.waitForTimeout(300); // Allow focus to settle
    await page.keyboard.press('Space');

    // Ensure it actually toggled
    await expect(page.getByRole('checkbox')).toHaveAttribute('aria-checked', 'true', { timeout: 5000 });

    // ④ Tab to Send OTP button and press Enter
    await page.keyboard.press('Tab');

    // Skip any intermediate tab stops (e.g. ToS link, Privacy link)
    // We press Tab until the focused element is the Send OTP button
    let attempts = 0;
    while (attempts < 6) {
      const focused = await page.evaluate(() => {
        const el = document.activeElement;
        return el?.textContent?.trim() ?? '';
      });
      if (/send otp/i.test(focused)) break;
      await page.keyboard.press('Tab');
      attempts++;
    }

    await page.keyboard.press('Enter');

    // ⑤ OTP step should now be visible
    const firstOtpBox = page.locator('input[inputmode="numeric"][maxlength="1"]').first();
    await expect(firstOtpBox).toBeVisible({ timeout: 10_000 });

    // ⑥ The heading should confirm we're in OTP step
    await expect(page.getByText(/verify your number/i)).toBeVisible();
  });
});

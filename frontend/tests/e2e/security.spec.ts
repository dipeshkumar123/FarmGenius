/**
 * Security E2E tests — FarmGenius
 *
 * Test IDs:
 *   TC-AUTH-009  All protected routes redirect to /login when unauthenticated
 *   TC-SEC-005   XSS payload in chat input is rendered as text, not executed
 */

import { test, expect } from '@playwright/test';
import { loginAsDemoFarmer } from './helpers/auth';

const BACKEND_BASE = 'http://localhost:8001';

// ─── TC-AUTH-009: Extended protected route check ─────────────────────────────

test.describe('TC-AUTH-009: All protected routes redirect to /login when unauthenticated', () => {
  const ALL_PROTECTED = ['/dashboard', '/chat', '/market', '/scan', '/schemes', '/weather'];

  for (const route of ALL_PROTECTED) {
    test(`${route} → /login when localStorage is empty`, async ({ page }) => {
      // Clear all storage before navigating
      await page.goto('/');
      await page.evaluate(() => {
        localStorage.clear();
        sessionStorage.clear();
      });

      await page.goto(route);
      await expect(page).toHaveURL(/\/login/, { timeout: 10_000 });
    });
  }
});

// ─── TC-SEC-005: XSS in chat input ───────────────────────────────────────────

test.describe('TC-SEC-005: XSS payload in chat input is rendered as text, not executed', () => {
  test('script tag in message is not executed', async ({ page }) => {
    // Track any unexpected dialog (alert / confirm / prompt)
    let dialogFired = false;
    page.on('dialog', async (dialog) => {
      dialogFired = true;
      await dialog.dismiss();
    });

    // Mock /chat so the XSS payload doesn't need a real backend
    await page.route(`${BACKEND_BASE}/chat`, (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          response: 'Here is safe advice for your crop.',
          source: 'Groq LLM',
          confidence: 0.9,
        }),
      });
    });

    // Capture the original page title
    await loginAsDemoFarmer(page);
    await page.goto('/chat');

    const originalTitle = await page.title();

    // Wait for chat to be ready
    const chatInput = page.locator('textarea[placeholder*="Ask anything"]');
    await chatInput.waitFor({ state: 'visible', timeout: 15_000 });

    // Send an XSS payload as a user message
    const xssPayload = '<script>document.title="XSS"</script>';
    await chatInput.fill(xssPayload);
    await chatInput.press('Enter');

    // Wait for bot response to arrive (confirms the message cycle completed)
    await page.locator('.bubble-bot').last().waitFor({ state: 'visible', timeout: 15_000 });

    // 1. Page title must NOT have been changed to "XSS"
    const currentTitle = await page.title();
    expect(currentTitle).not.toBe('XSS');

    // 2. The title should still match the original (or at minimum not be "XSS")
    expect(currentTitle).toBe(originalTitle);

    // 3. No dialog (alert) must have fired
    expect(dialogFired).toBe(false);

    // 4. The raw payload text should be visible as text in the user bubble
    //    (React escapes innerHTML, so it renders as literal characters)
    const userBubble = page.locator('.bubble-user').last();
    await expect(userBubble).toContainText('<script>', { timeout: 5_000 });
  });
});

/**
 * Chat E2E tests — FarmGenius Chat Page
 *
 * Test IDs:
 *   TC-CHAT-001  Send a message and receive AI response
 *   TC-CHAT-003  Chat shows error when backend fails
 */

import { test, expect } from '@playwright/test';
import { loginAsDemoFarmer } from './helpers/auth';

const BACKEND_BASE = 'http://localhost:8001';
const CHAT_ENDPOINT = `${BACKEND_BASE}/chat`;

// ─── Tests ────────────────────────────────────────────────────────────────────

test.describe('TC-CHAT-001: Send a message and receive AI response', () => {
  test('user message appears and AI response bubble is shown', async ({ page }) => {
    // ① Mock /auth/verify-otp for a quick demo login path (used by some navigation flows)
    await page.route(`${BACKEND_BASE}/auth/verify-otp`, (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ token: 'mock_jwt_token', farmer_id: '9999999999' }),
      });
    });

    // ② Mock /chat to return a deterministic AI response
    await page.route(CHAT_ENDPOINT, (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          response: 'Use Azoxystrobin fungicide. Source: KVK Karnataka',
          source: 'Groq LLM',
          confidence: 0.95,
        }),
      });
    });

    // ③ Bypass UI login via localStorage injection
    await loginAsDemoFarmer(page);

    // ④ Navigate to /chat
    await page.goto('/chat');

    // ⑤ Wait for the chat textarea to be visible
    const chatInput = page.locator('textarea[placeholder*="Ask anything"]');
    await chatInput.waitFor({ state: 'visible', timeout: 15_000 });

    // ⑥ Type a message and press Enter
    const userMessage = 'My tomato leaves have white powder, what disease is this?';
    await chatInput.fill(userMessage);
    await chatInput.press('Enter');

    // ⑦ Assert user message bubble appears (sender='user' → .bubble-user)
    const userBubble = page.locator('.bubble-user').filter({ hasText: userMessage });
    await expect(userBubble).toBeVisible({ timeout: 10_000 });

    // ⑧ Assert AI response bubble appears with the mocked text
    const botBubble = page.locator('.bubble-bot').filter({
      hasText: 'Use Azoxystrobin fungicide',
    });
    await expect(botBubble).toBeVisible({ timeout: 15_000 });
  });
});

test.describe('TC-CHAT-003: Chat shows error when backend fails', () => {
  test('500 from /chat — app falls back gracefully and shows a message', async ({ page }) => {
    // Mock /chat to return 500
    await page.route(CHAT_ENDPOINT, (route) => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Internal Server Error' }),
      });
    });

    await loginAsDemoFarmer(page);
    await page.goto('/chat');

    const chatInput = page.locator('textarea[placeholder*="Ask anything"]');
    await chatInput.waitFor({ state: 'visible', timeout: 15_000 });

    await chatInput.fill('What crops should I grow?');
    await chatInput.press('Enter');

    // User message should still appear
    const userBubble = page.locator('.bubble-user').filter({
      hasText: 'What crops should I grow?',
    });
    await expect(userBubble).toBeVisible({ timeout: 10_000 });

    // The app falls back to mock responses on error, so a bot bubble should appear
    // We wait for a bot bubble that actually contains text (bypassing the empty typing indicator)
    const botBubble = page.locator('.bubble-bot').filter({ hasText: /[a-zA-Z]/ }).last();
    await expect(botBubble).toBeVisible({ timeout: 15_000 });

    // The fallback response should NOT be empty
    const botText = await botBubble.textContent();
    expect(botText?.trim().length).toBeGreaterThan(0);
  });
});

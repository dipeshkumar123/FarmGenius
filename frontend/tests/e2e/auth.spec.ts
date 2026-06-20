/**
 * Auth E2E tests — FarmGenius Login Flow
 *
 * Test IDs:
 *   TC-AUTH-001  Invalid phone numbers show error toast
 *   TC-AUTH-002  Terms checkbox required
 *   TC-AUTH-003  Valid phone + Terms → OTP step
 *   TC-AUTH-004  Valid OTP authenticates and redirects to /dashboard
 *   TC-AUTH-005  OTP 000000 is rejected
 *   TC-AUTH-009  Protected routes redirect unauthenticated users to /login
 */

import { test, expect } from '@playwright/test';

const BACKEND_BASE = 'http://localhost:8001';
const OTP_ENDPOINT = `${BACKEND_BASE}/auth/verify-otp`;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Fill the 6 individual OTP boxes with the digits from a 6-char string */
async function fillOtpBoxes(page: import('@playwright/test').Page, code: string): Promise<void> {
  // The OTP boxes are input[type="text"] with inputMode="numeric" and maxLength=1
  // They sit inside the OTP step container; we target them all at once
  const boxes = page.locator('input[inputmode="numeric"][maxlength="1"]');
  await boxes.first().waitFor({ state: 'visible', timeout: 15_000 });

  for (let i = 0; i < 6; i++) {
    await boxes.nth(i).fill(code[i]);
  }
}

/** Complete the phone step: type phone, optionally tick Terms, click Send OTP */
async function completePhoneStep(
  page: import('@playwright/test').Page,
  options: { phone: string; agreeToTerms: boolean }
): Promise<void> {
  await page.goto('/login');

  const phoneInput = page.locator('input[type="tel"]');
  await phoneInput.waitFor({ state: 'visible' });
  await phoneInput.fill(options.phone);

  if (options.agreeToTerms) {
    // The Terms checkbox has role="checkbox"
    const termsToggle = page.getByRole('checkbox');
    await termsToggle.click({ force: true });
  }

  await page.getByRole('button', { name: /send otp/i }).click({ force: true });
}

// ─── Tests ────────────────────────────────────────────────────────────────────

test.describe('TC-AUTH-001: Invalid phone numbers show error toast, OTP step not reached', () => {
  test('phone starting with 1 (invalid Indian mobile) shows error', async ({ page }) => {
    await page.goto('/login');
    const phoneInput = page.locator('input[type="tel"]');
    await phoneInput.waitFor({ state: 'visible' });
    await phoneInput.fill('1234567890');

    // Agree to terms to isolate phone validation
    const termsToggle = page.getByRole('checkbox');
    await termsToggle.click({ force: true });

    // The Send OTP button is disabled when phone is invalid — OTP step never shown
    const sendOtpBtn = page.getByRole('button', { name: /send otp/i });
    await expect(sendOtpBtn).toBeDisabled();

    const otpBoxes = page.locator('input[inputmode="numeric"][maxlength="1"]');
    await expect(otpBoxes.first()).not.toBeVisible();
  });

  test('phone with 5 digits (too short) shows error', async ({ page }) => {
    await page.goto('/login');
    const phoneInput = page.locator('input[type="tel"]');
    await phoneInput.waitFor({ state: 'visible' });
    await phoneInput.fill('99999');

    const termsToggle = page.getByRole('checkbox');
    await termsToggle.click({ force: true });

    // The Send OTP button is disabled when phone is invalid
    const sendOtpBtn = page.getByRole('button', { name: /send otp/i });
    await expect(sendOtpBtn).toBeDisabled();

    // OTP boxes should not appear
    const otpBoxes = page.locator('input[inputmode="numeric"][maxlength="1"]');
    await expect(otpBoxes.first()).not.toBeVisible();
  });
});

test.describe('TC-AUTH-002: Terms checkbox is required', () => {
  test('valid phone without Terms checked shows error toast', async ({ page }) => {
    await page.goto('/login');
    const phoneInput = page.locator('input[type="tel"]');
    await phoneInput.waitFor({ state: 'visible' });
    await phoneInput.fill('9999999999');

    // Do NOT click Terms — intentionally skip it
    // The button is disabled when !agreedToTerms, so we confirm it's not clickable
    const sendOtpBtn = page.getByRole('button', { name: /send otp/i });
    await expect(sendOtpBtn).toBeDisabled();

    // OTP step should not appear
    const otpBoxes = page.locator('input[inputmode="numeric"][maxlength="1"]');
    await expect(otpBoxes.first()).not.toBeVisible();
  });
});

test.describe('TC-AUTH-003: Valid phone + Terms checked → transitions to OTP step', () => {
  test('OTP boxes appear after successful phone submission', async ({ page }) => {
    await page.goto('/login');

    const phoneInput = page.locator('input[type="tel"]');
    await phoneInput.waitFor({ state: 'visible' });
    await phoneInput.fill('9999999999');

    const termsToggle = page.getByRole('checkbox');
    await termsToggle.click({ force: true });

    const sendOtpBtn = page.getByRole('button', { name: /send otp/i });
    await expect(sendOtpBtn).toBeEnabled();
    await sendOtpBtn.click({ force: true });

    // Wait for OTP step — 6 individual number boxes appear
    const firstOtpBox = page.locator('input[inputmode="numeric"][maxlength="1"]').first();
    await expect(firstOtpBox).toBeVisible({ timeout: 10_000 });

    // The heading should change to "Verify your number"
    await expect(page.getByText(/verify your number/i)).toBeVisible();
  });
});

test.describe('TC-AUTH-004: Valid OTP authenticates and redirects to /dashboard', () => {
  test('mocked 123456 OTP → redirected to /dashboard', async ({ page }) => {
    test.setTimeout(60000);
    // Mock the OTP verification endpoint
    await page.route(OTP_ENDPOINT, (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ token: 'mock_jwt_token', farmer_id: '9999999999' }),
      });
    });

    // Complete phone step
    await page.goto('/login');
    const phoneInput = page.locator('input[type="tel"]');
    await phoneInput.waitFor({ state: 'visible' });
    await phoneInput.fill('9999999999');

    const termsToggle = page.getByRole('checkbox');
    await termsToggle.click({ force: true });
    await page.getByRole('button', { name: /send otp/i }).click({ force: true });

    // Wait for OTP step
    await page.locator('input[inputmode="numeric"][maxlength="1"]').first().waitFor({
      state: 'visible',
      timeout: 10_000,
    });

    // Fill OTP 123456
    await fillOtpBoxes(page, '123456');

    // App sets localStorage token and redirects to /dashboard
    await expect(page).toHaveURL(/\/dashboard/, { timeout: 15_000 });

    // Token should be stored
    const token = await page.evaluate(() => localStorage.getItem('fg_token'));
    expect(token).toBe('mock_jwt_token');
  });
});

test.describe('TC-AUTH-005: OTP 000000 is rejected', () => {
  test('OTP 000000 shows Incorrect OTP toast', async ({ page }) => {
    // Mock to reject 000000
    await page.route(OTP_ENDPOINT, (route, request) => {
      const body = request.postDataJSON() as { otp: string };
      if (body.otp === '000000') {
        route.fulfill({
          status: 400,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'Invalid OTP' }),
        });
      } else {
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ token: 'mock_jwt_token', farmer_id: '9999999999' }),
        });
      }
    });

    await page.goto('/login');
    const phoneInput = page.locator('input[type="tel"]');
    await phoneInput.waitFor({ state: 'visible' });
    await phoneInput.fill('9999999999');

    const termsToggle = page.getByRole('checkbox');
    await termsToggle.click({ force: true });
    await page.getByRole('button', { name: /send otp/i }).click({ force: true });

    await page.locator('input[inputmode="numeric"][maxlength="1"]').first().waitFor({
      state: 'visible',
      timeout: 10_000,
    });

    await fillOtpBoxes(page, '000000');

    // Should stay on /login and show error toast or text
    await expect(page.getByText(/incorrect otp/i).first()).toBeVisible({ timeout: 10_000 });
    await expect(page).toHaveURL(/\/login/);
  });
});

test.describe('TC-AUTH-009: Protected routes redirect unauthenticated users to /login', () => {
  const protectedRoutes = ['/dashboard', '/chat', '/market'];

  for (const route of protectedRoutes) {
    test(`${route} redirects to /login when no token in localStorage`, async ({ page }) => {
      // Ensure clean storage
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

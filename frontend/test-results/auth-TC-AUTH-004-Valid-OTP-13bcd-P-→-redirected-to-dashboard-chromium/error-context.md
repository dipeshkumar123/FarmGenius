# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: auth.spec.ts >> TC-AUTH-004: Valid OTP authenticates and redirects to /dashboard >> mocked 123456 OTP → redirected to /dashboard
- Location: tests\e2e\auth.spec.ts:135:3

# Error details

```
Error: page.goto: net::ERR_CONNECTION_REFUSED at http://localhost:5173/login
Call log:
  - navigating to "http://localhost:5173/login", waiting until "load"

```

# Test source

```ts
  47  |   }
  48  | 
  49  |   await page.getByRole('button', { name: /send otp/i }).click({ force: true });
  50  | }
  51  | 
  52  | // ─── Tests ────────────────────────────────────────────────────────────────────
  53  | 
  54  | test.describe('TC-AUTH-001: Invalid phone numbers show error toast, OTP step not reached', () => {
  55  |   test('phone starting with 1 (invalid Indian mobile) shows error', async ({ page }) => {
  56  |     await page.goto('/login');
  57  |     const phoneInput = page.locator('input[type="tel"]');
  58  |     await phoneInput.waitFor({ state: 'visible' });
  59  |     await phoneInput.fill('1234567890');
  60  | 
  61  |     // Agree to terms to isolate phone validation
  62  |     const termsToggle = page.getByRole('checkbox');
  63  |     await termsToggle.click({ force: true });
  64  | 
  65  |     // The Send OTP button is disabled when phone is invalid — OTP step never shown
  66  |     const sendOtpBtn = page.getByRole('button', { name: /send otp/i });
  67  |     await expect(sendOtpBtn).toBeDisabled();
  68  | 
  69  |     const otpBoxes = page.locator('input[inputmode="numeric"][maxlength="1"]');
  70  |     await expect(otpBoxes.first()).not.toBeVisible();
  71  |   });
  72  | 
  73  |   test('phone with 5 digits (too short) shows error', async ({ page }) => {
  74  |     await page.goto('/login');
  75  |     const phoneInput = page.locator('input[type="tel"]');
  76  |     await phoneInput.waitFor({ state: 'visible' });
  77  |     await phoneInput.fill('99999');
  78  | 
  79  |     const termsToggle = page.getByRole('checkbox');
  80  |     await termsToggle.click({ force: true });
  81  | 
  82  |     // The Send OTP button is disabled when phone is invalid
  83  |     const sendOtpBtn = page.getByRole('button', { name: /send otp/i });
  84  |     await expect(sendOtpBtn).toBeDisabled();
  85  | 
  86  |     // OTP boxes should not appear
  87  |     const otpBoxes = page.locator('input[inputmode="numeric"][maxlength="1"]');
  88  |     await expect(otpBoxes.first()).not.toBeVisible();
  89  |   });
  90  | });
  91  | 
  92  | test.describe('TC-AUTH-002: Terms checkbox is required', () => {
  93  |   test('valid phone without Terms checked shows error toast', async ({ page }) => {
  94  |     await page.goto('/login');
  95  |     const phoneInput = page.locator('input[type="tel"]');
  96  |     await phoneInput.waitFor({ state: 'visible' });
  97  |     await phoneInput.fill('9999999999');
  98  | 
  99  |     // Do NOT click Terms — intentionally skip it
  100 |     // The button is disabled when !agreedToTerms, so we confirm it's not clickable
  101 |     const sendOtpBtn = page.getByRole('button', { name: /send otp/i });
  102 |     await expect(sendOtpBtn).toBeDisabled();
  103 | 
  104 |     // OTP step should not appear
  105 |     const otpBoxes = page.locator('input[inputmode="numeric"][maxlength="1"]');
  106 |     await expect(otpBoxes.first()).not.toBeVisible();
  107 |   });
  108 | });
  109 | 
  110 | test.describe('TC-AUTH-003: Valid phone + Terms checked → transitions to OTP step', () => {
  111 |   test('OTP boxes appear after successful phone submission', async ({ page }) => {
  112 |     await page.goto('/login');
  113 | 
  114 |     const phoneInput = page.locator('input[type="tel"]');
  115 |     await phoneInput.waitFor({ state: 'visible' });
  116 |     await phoneInput.fill('9999999999');
  117 | 
  118 |     const termsToggle = page.getByRole('checkbox');
  119 |     await termsToggle.click({ force: true });
  120 | 
  121 |     const sendOtpBtn = page.getByRole('button', { name: /send otp/i });
  122 |     await expect(sendOtpBtn).toBeEnabled();
  123 |     await sendOtpBtn.click({ force: true });
  124 | 
  125 |     // Wait for OTP step — 6 individual number boxes appear
  126 |     const firstOtpBox = page.locator('input[inputmode="numeric"][maxlength="1"]').first();
  127 |     await expect(firstOtpBox).toBeVisible({ timeout: 10_000 });
  128 | 
  129 |     // The heading should change to "Verify your number"
  130 |     await expect(page.getByText(/verify your number/i)).toBeVisible();
  131 |   });
  132 | });
  133 | 
  134 | test.describe('TC-AUTH-004: Valid OTP authenticates and redirects to /dashboard', () => {
  135 |   test('mocked 123456 OTP → redirected to /dashboard', async ({ page }) => {
  136 |     test.setTimeout(60000);
  137 |     // Mock the OTP verification endpoint
  138 |     await page.route(OTP_ENDPOINT, (route) => {
  139 |       route.fulfill({
  140 |         status: 200,
  141 |         contentType: 'application/json',
  142 |         body: JSON.stringify({ token: 'mock_jwt_token', farmer_id: '9999999999' }),
  143 |       });
  144 |     });
  145 | 
  146 |     // Complete phone step
> 147 |     await page.goto('/login');
      |                ^ Error: page.goto: net::ERR_CONNECTION_REFUSED at http://localhost:5173/login
  148 |     const phoneInput = page.locator('input[type="tel"]');
  149 |     await phoneInput.waitFor({ state: 'visible' });
  150 |     await phoneInput.fill('9999999999');
  151 | 
  152 |     const termsToggle = page.getByRole('checkbox');
  153 |     await termsToggle.click({ force: true });
  154 |     await page.getByRole('button', { name: /send otp/i }).click({ force: true });
  155 | 
  156 |     // Wait for OTP step
  157 |     await page.locator('input[inputmode="numeric"][maxlength="1"]').first().waitFor({
  158 |       state: 'visible',
  159 |       timeout: 10_000,
  160 |     });
  161 | 
  162 |     // Fill OTP 123456
  163 |     await fillOtpBoxes(page, '123456');
  164 | 
  165 |     // App sets localStorage token and redirects to /dashboard
  166 |     await expect(page).toHaveURL(/\/dashboard/, { timeout: 15_000 });
  167 | 
  168 |     // Token should be stored
  169 |     const token = await page.evaluate(() => localStorage.getItem('fg_token'));
  170 |     expect(token).toBe('mock_jwt_token');
  171 |   });
  172 | });
  173 | 
  174 | test.describe('TC-AUTH-005: OTP 000000 is rejected', () => {
  175 |   test('OTP 000000 shows Incorrect OTP toast', async ({ page }) => {
  176 |     // Mock to reject 000000
  177 |     await page.route(OTP_ENDPOINT, (route, request) => {
  178 |       const body = request.postDataJSON() as { otp: string };
  179 |       if (body.otp === '000000') {
  180 |         route.fulfill({
  181 |           status: 400,
  182 |           contentType: 'application/json',
  183 |           body: JSON.stringify({ detail: 'Invalid OTP' }),
  184 |         });
  185 |       } else {
  186 |         route.fulfill({
  187 |           status: 200,
  188 |           contentType: 'application/json',
  189 |           body: JSON.stringify({ token: 'mock_jwt_token', farmer_id: '9999999999' }),
  190 |         });
  191 |       }
  192 |     });
  193 | 
  194 |     await page.goto('/login');
  195 |     const phoneInput = page.locator('input[type="tel"]');
  196 |     await phoneInput.waitFor({ state: 'visible' });
  197 |     await phoneInput.fill('9999999999');
  198 | 
  199 |     const termsToggle = page.getByRole('checkbox');
  200 |     await termsToggle.click({ force: true });
  201 |     await page.getByRole('button', { name: /send otp/i }).click({ force: true });
  202 | 
  203 |     await page.locator('input[inputmode="numeric"][maxlength="1"]').first().waitFor({
  204 |       state: 'visible',
  205 |       timeout: 10_000,
  206 |     });
  207 | 
  208 |     await fillOtpBoxes(page, '000000');
  209 | 
  210 |     // Should stay on /login and show error toast or text
  211 |     await expect(page.getByText(/incorrect otp/i).first()).toBeVisible({ timeout: 10_000 });
  212 |     await expect(page).toHaveURL(/\/login/);
  213 |   });
  214 | });
  215 | 
  216 | test.describe('TC-AUTH-009: Protected routes redirect unauthenticated users to /login', () => {
  217 |   const protectedRoutes = ['/dashboard', '/chat', '/market'];
  218 | 
  219 |   for (const route of protectedRoutes) {
  220 |     test(`${route} redirects to /login when no token in localStorage`, async ({ page }) => {
  221 |       // Ensure clean storage
  222 |       await page.goto('/');
  223 |       await page.evaluate(() => {
  224 |         localStorage.clear();
  225 |         sessionStorage.clear();
  226 |       });
  227 | 
  228 |       await page.goto(route);
  229 |       await expect(page).toHaveURL(/\/login/, { timeout: 10_000 });
  230 |     });
  231 |   }
  232 | });
  233 | 
```
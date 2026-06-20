import type { Page } from '@playwright/test';

/**
 * Bypass the UI login flow by injecting a mock JWT into localStorage.
 * Also writes the Zustand 'farmgenius-app' key so the app thinks the
 * user is authenticated without hitting the real backend.
 */
export async function loginAsDemoFarmer(page: Page): Promise<void> {
  // Navigate to the app root so we have a same-origin context to set storage
  await page.goto('/');

  await page.evaluate(() => {
    // Set the raw JWT that the API client reads from
    const mockToken =
      'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.' +
      'eyJzdWIiOiI5OTk5OTk5OTk5IiwiaWF0IjoxNzAwMDAwMDAwLCJleHAiOjk5OTk5OTk5OTl9.' +
      'MOCK_SIGNATURE';
    localStorage.setItem('fg_token', mockToken);

    // Hydrate Zustand persisted state so ProtectedRoute passes
    const zustandState = {
      state: {
        isAuthenticated: true,
        language: 'en',
        isOffline: false,
        farmer: {
          name: 'Demo Farmer',
          phone: '9999999999',
          district: 'Dharwad',
          state: 'Karnataka',
          language: 'en',
          crops: ['Wheat', 'Tomato', 'Onion'],
        },
      },
      version: 0,
    };
    localStorage.setItem('farmgenius-app', JSON.stringify(zustandState));
  });
}

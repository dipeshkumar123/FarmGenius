const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  const errors = [];
  const networkErrors = [];

  page.on('console', msg => {
    if (msg.type() === 'error') {
      errors.push(`[Console Error] ${msg.text()}`);
    }
  });

  page.on('requestfailed', request => {
    networkErrors.push(`[Network Error] ${request.url()} - ${request.failure().errorText}`);
  });

  page.on('response', response => {
    if (!response.ok()) {
      networkErrors.push(`[HTTP Error] ${response.status()} ${response.url()}`);
    }
  });

  console.log('Navigating to https://farmgenius-monorepo.vercel.app/');
  
  try {
    await page.goto('https://farmgenius-monorepo.vercel.app/', { waitUntil: 'networkidle' });
    console.log('Page loaded.');
    await page.waitForTimeout(2000);

    console.log('--- Errors Captured on Load ---');
    console.log('Console Errors:', errors);
    console.log('Network Errors:', networkErrors);

    // Let's try to login
    console.log('Attempting Login...');
    const phoneInput = await page.$('input[placeholder*="10-digit"]');
    if (phoneInput) {
      await phoneInput.fill('9999999999');
      
      const termsCheckbox = await page.$('input[type="checkbox"]');
      if (termsCheckbox) await termsCheckbox.check();

      const sendOtpBtn = await page.getByRole('button', { name: /Send OTP/i });
      if (sendOtpBtn) {
        await sendOtpBtn.click();
        await page.waitForTimeout(2000);

        // Enter OTP 123456
        const otpInputs = await page.$$('input[type="text"]');
        for (let i = 0; i < Math.min(6, otpInputs.length); i++) {
          await otpInputs[i].fill((i + 1).toString());
        }
        
        const verifyBtn = await page.getByRole('button', { name: /Verify/i });
        if (verifyBtn) {
          await verifyBtn.click();
          await page.waitForTimeout(4000);
        }
      }
    } else {
      console.log('No phone input found, maybe already logged in or on dashboard?');
    }

    console.log('--- Errors Captured After Login ---');
    console.log('Console Errors:', errors);
    console.log('Network Errors:', networkErrors);

    console.log('Current URL:', page.url());

    // Try to visit pages
    const pagesToVisit = ['/dashboard', '/chat', '/scan', '/weather', '/market', '/schemes', '/profile'];
    for (const route of pagesToVisit) {
      console.log(`\nNavigating to ${route}...`);
      await page.goto(`https://farmgenius-monorepo.vercel.app${route}`, { waitUntil: 'networkidle' });
      await page.waitForTimeout(2000);
      console.log(`URL after nav: ${page.url()}`);
    }

    console.log('\n--- Final Error Summary ---');
    console.log('Console Errors:', [...new Set(errors)]);
    console.log('Network Errors:', [...new Set(networkErrors)]);
    
  } catch (e) {
    console.error('Test script crashed:', e);
  } finally {
    await browser.close();
  }
})();

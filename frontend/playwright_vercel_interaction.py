import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        console_messages = []
        page.on("console", lambda msg: console_messages.append(f"[{msg.type}] {msg.text}"))
        
        page_errors = []
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        print("Navigating to URL...")
        # Go straight to vercel
        await page.goto("https://frontend-pi-liart-51.vercel.app/login", wait_until="networkidle")
        await page.wait_for_timeout(1000)

        print("Typing phone number...")
        await page.fill('input[type="tel"]', '9876543210')
        
        print("Checking terms...")
        # Click the actual div that toggles it
        await page.locator('label div.w-5.h-5').click()
        
        print("Clicking send OTP...")
        await page.click('button:has-text("Send OTP")')
        await page.wait_for_timeout(2000)
        
        print("Entering OTP...")
        # Type OTP
        await page.keyboard.type("123456")
        await page.wait_for_timeout(2000)

        with open("playwright_interaction_logs.txt", "w", encoding="utf-8") as f:
            f.write("=== CONSOLE LOGS ===\n")
            for msg in console_messages:
                f.write(msg + "\n")
            f.write("\n=== PAGE ERRORS ===\n")
            for err in page_errors:
                f.write(err + "\n")

        await browser.close()

asyncio.run(main())

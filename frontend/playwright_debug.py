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
        await page.goto("https://frontend-pi-liart-51.vercel.app/login", wait_until="networkidle")
        await page.wait_for_timeout(3000)

        # take a screenshot
        await page.screenshot(path="playwright_screenshot.png")
        content = await page.content()

        with open("playwright_logs.txt", "w", encoding="utf-8") as f:
            f.write("=== CONSOLE LOGS ===\n")
            for msg in console_messages:
                f.write(msg + "\n")
            f.write("\n=== PAGE ERRORS ===\n")
            for err in page_errors:
                f.write(err + "\n")
            f.write("\n=== CONTENT ===\n")
            f.write(content)

        await browser.close()

asyncio.run(main())

import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import json

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

url = "https://frontend-pi-liart-51.vercel.app/login"
driver.get(url)

time.sleep(5)

logs = driver.get_log("browser")
with open("selenium_logs.txt", "w", encoding="utf-8") as f:
    f.write(f"Page title: {driver.title}\n")
    f.write("Console logs:\n")
    for log in logs:
        f.write(f"[{log['level']}] {log['message']}\n")
    f.write("\nPage source:\n")
    f.write(driver.page_source)

driver.quit()

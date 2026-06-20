from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

options = Options()
options.add_argument('--headless=new')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920,1080')

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)
driver.get('https://frontend-pi-liart-51.vercel.app/login')
time.sleep(10)

print('\n--- CONSOLE LOGS ---')
for entry in driver.get_log('browser'):
    print(entry)
print('--------------------\n')

driver.save_screenshot('frontend_screenshot.png')
driver.quit()

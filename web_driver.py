import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Use undetected-chromedriver to bypass automation detection
driver = uc.Chrome()

driver.get("https://library.bu.edu/netadvantage")
time.sleep(2)

username_input = driver.find_element(By.ID, "j_username")  # Target the ID 'j_username'
username_input.clear()
username_input.send_keys("email")  # Replace 'your_username' with your actual BU email

# Step 4: Find the password input field and enter the password
password_input = driver.find_element(By.ID, "j_password")  # Target the ID 'j_password'
password_input.clear()
password_input.send_keys("password")  # Replace 'your_password' with your actual password

# Step 5: Submit the form (usually either pressing Enter or finding a login button)
password_input.send_keys(Keys.RETURN)

time.sleep(5)

send_button = driver.find_element(By.CLASS_NAME, "button--primary--full-width")
send_button.click()

time.sleep(5)

sms_input = driver.find_element(By.ID, "passcode-input")
sms_input.clear()

# Step 3: Manually retrieve the passcode from your phone and enter it
sms_code = "123456"  
sms_input.send_keys(sms_code)

# Step 4: Submit the passcode form (if needed)
sms_input.send_keys(Keys.RETURN)

time.sleep(40)

driver.close()
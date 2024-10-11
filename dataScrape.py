import pyautogui
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import ssl

# Disable SSL certificate verification (globally)
ssl._create_default_https_context = ssl._create_unverified_context

def _setup_driver():
    """
    Private method to set up the Chrome WebDriver with options and returns the driver object.
    """
    chrome_options = Options()
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--allow-insecure-localhost')
    chrome_options.add_argument('--allow-running-insecure-content')

    # Use webdriver-manager to download and manage the correct ChromeDriver version
    service = Service(ChromeDriverManager().install())
    
    # Initialize Chrome with options
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def _login(driver, username, password):
    """
    Private method to log in to the page using provided credentials.
    """
    username_input = driver.find_element(By.ID, "j_username")
    username_input.clear()
    username_input.send_keys(username)
    
    password_input = driver.find_element(By.ID, "j_password")
    password_input.clear()
    password_input.send_keys(password)
    
    password_input.send_keys(Keys.RETURN)
    time.sleep(2)

def _handle_passkey_prompt():
    """
    Private method to handle macOS passkey authentication prompt using pyautogui.
    """
    try:
        time.sleep(2)
        pyautogui.press('enter')  # Press "Enter" to click "Continue"
        time.sleep(2)
        system_password = "123456"  # Replace with your actual macOS system password
        pyautogui.write(system_password, interval=0.1)
        pyautogui.press('enter')
    except Exception as e:
        print(f"Error handling passkey prompt: {e}")

def _click_trust_device(driver):
    """
    Private method to click the 'Yes, this is my device' button.
    """
    try:
        trust_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "trust-browser-button"))
        )
        trust_button.click()
        time.sleep(2)
    except Exception as e:
        print(f"Error clicking 'Yes, this is my device': {e}")

def _search_ticker(driver, ticker_symbol):
    """
    Private method to search for a ticker symbol using the search input field.
    """
    try:
        search_input = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "SearchTopBar"))
        )
        search_input.clear()
        search_input.send_keys(ticker_symbol)
        search_input.send_keys(Keys.RETURN)
        time.sleep(2)
    except Exception as e:
        print(f"Error while searching for the ticker: {e}")

def _click_key_developments(driver):
    """
    Private method to click the 'Key Developments' link on the page.
    """
    try:
        key_developments_link = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.ID, "ll_7_10047_812_my"))
        )
        key_developments_link.click()
        time.sleep(2)
    except Exception as e:
        print(f"Error while clicking 'Key Developments': {e}")

def _click_expand_icon(driver):
    """
    Private method to click the "+" icon to expand all rows in the Key Developments section.
    """
    try:
        expand_icon = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.ID, "Displaysection3_myKeyDevDataGrid_myDataGrid_Icon"))
        )
        expand_icon.click()
        print("Clicked the '+' icon to expand rows.")
    except Exception as e:
        print(f"Error clicking '+' icon: {e}")

def _extract_all_data(driver):
    """
    Private method to extract and print all 'Type' and 'Situation' data from the Key Developments section,
    and return them as a DataFrame.
    """
    try:
        situation_elements = WebDriverWait(driver, 5).until(
            EC.presence_of_all_elements_located((By.XPATH, "/html/body/table/tbody/tr[2]/td[4]/div/form/div[5]/table/tbody/tr/td/span/table[1]/tbody/tr/td[2]/div/table/tbody/tr/td"))
        )
        type_elements = WebDriverWait(driver, 5).until(
            EC.presence_of_all_elements_located((By.XPATH, "//td[@align='left' and @valign='top' and @style='width:200px;']/span[1]"))
        )

        situations = [element.text.strip() for element in situation_elements if element.text.strip()]
        types = [element.text.strip() for element in type_elements if element.text.strip()]

        if len(situations) == len(types):
            print("Situations and types extracted successfully.")
        else:
            print("Warning: Mismatch in number of situations and types.")

        data = {
            'Type': types[:len(situations)],
            'Situation': situations
        }
        df = pd.DataFrame(data)
        print(df)

        return df

    except Exception as e:
        print(f"Error extracting data: {e}")
        return None

def main(username, password, ticker):
    """
    Main function to scrape data after logging in and interacting with the website.
    Returns the extracted DataFrame.
    """
    driver = _setup_driver()

    # Navigate to the login page
    driver.get("https://library.bu.edu/netadvantage")
    time.sleep(2)

    # Login
    _login(driver, username, password)

    # Handle any potential passkey prompt
    _handle_passkey_prompt()

    # Click trust device button
    _click_trust_device(driver)

    # Search for the ticker symbol
    _search_ticker(driver, ticker)

    # Navigate and expand key developments
    _click_key_developments(driver)
    _click_expand_icon(driver)

    # Extract data
    df = _extract_all_data(driver)

    time.sleep()
    # Close the driver
    driver.quit()

    return df

# Example of calling the main method from another script
if __name__ == "__main__":
    username = "atharvam"  # Replace with your actual BU username
    password = "QpAlZmwoskxn@00001"  # Replace with your actual BU password
    ticker = "AAPL"  # Replace with your desired ticker symbol
    data = main(username, password, ticker)
    print(data)

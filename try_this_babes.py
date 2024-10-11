import pyautogui
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

class SeleniumScraper:
    def __init__(self, username, password, system_password, ticker_symbol):
        """
        Initialize the SeleniumScraper class with login credentials and the ticker symbol to search for.
        """
        self.username = username
        self.password = password
        self.system_password = system_password
        self.ticker_symbol = ticker_symbol
        
        # Disable SSL certificate verification (globally)
        ssl._create_default_https_context = ssl._create_unverified_context

        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--allow-insecure-localhost')
        chrome_options.add_argument('--allow-running-insecure-content')

        # Use webdriver-manager to download and manage the correct ChromeDriver version
        self.service = Service(ChromeDriverManager().install())

        # Initialize Chrome with options
        self.driver = webdriver.Chrome(service=self.service, options=chrome_options)

    def navigate_to_page(self, url):
        """
        Navigate to the specified page.
        """
        self.driver.get(url)
        time.sleep(2)  # Wait for the page to load

    def login(self):
        """
        Log in to the website using the provided credentials.
        """
        try:
            username_input = self.driver.find_element(By.ID, "j_username")
            username_input.clear()
            username_input.send_keys(self.username)
            
            password_input = self.driver.find_element(By.ID, "j_password")
            password_input.clear()
            password_input.send_keys(self.password)
            
            password_input.send_keys(Keys.RETURN)  # Submit the form
            time.sleep(2)
        except Exception as e:
            print(f"Error during login: {e}")

    def handle_passkey_prompt(self):
        """
        Handle macOS passkey authentication prompt using pyautogui.
        """
        try:
            time.sleep(2)  # Wait for the passkey prompt
            pyautogui.press('enter')  # Press "Enter" to click "Continue"
            time.sleep(2)  # Wait for the password prompt
            
            pyautogui.write(self.system_password, interval=0.1)  # Type the system password
            pyautogui.press('enter')  # Submit the password
        except Exception as e:
            print(f"Error handling passkey prompt: {e}")

    def click_trust_device(self):
        """
        Click the 'Yes, this is my device' button.
        """
        try:
            trust_button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.ID, "trust-browser-button"))
            )
            trust_button.click()
            time.sleep(2)
        except Exception as e:
            print(f"Error clicking 'Yes, this is my device': {e}")

    def search_ticker(self):
        """
        Search for a company ticker symbol.
        """
        try:
            search_input = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.ID, "SearchTopBar"))
            )
            search_input.clear()
            search_input.send_keys(self.ticker_symbol)
            search_input.send_keys(Keys.RETURN)  # Submit the search
            time.sleep(2)
        except Exception as e:
            print(f"Error while searching for the ticker: {e}")

    def click_key_developments(self):
        """
        Click the 'Key Developments' link on the page.
        """
        try:
            key_developments_link = WebDriverWait(self.driver, 3).until(
                EC.element_to_be_clickable((By.ID, "ll_7_10047_812_my"))
            )
            key_developments_link.click()
            time.sleep(2)
        except Exception as e:
            print(f"Error while clicking 'Key Developments': {e}")

    def click_expand_icon(self):
        """
        Click the "+" icon to expand all rows in the Key Developments section.
        """
        try:
            expand_icon = WebDriverWait(self.driver, 3).until(
                EC.element_to_be_clickable((By.ID, "Displaysection3_myKeyDevDataGrid_myDataGrid_Icon"))
            )
            expand_icon.click()
            print("Clicked the '+' icon to expand rows.")
        except Exception as e:
            print(f"Error clicking '+' icon: {e}")

    def extract_all_situations(self):
        """
        Extract all 'Situation' data from the Key Developments section.
        """
        try:
            situation_elements = WebDriverWait(self.driver, 5).until(
                EC.presence_of_all_elements_located((By.XPATH, "/html/body/table/tbody/tr[2]/td[4]/div/form/div[5]/table/tbody/tr/td/span/table[1]/tbody/tr/td[2]/div/table/tbody/tr/td"))
            )

            situations = [element.text.strip() for element in situation_elements if element.text.strip()]
            print("All situations extracted:")
            for situation in situations:
                print(situation)

            with open('all_situations.txt', 'w') as f:
                for situation in situations:
                    f.write(f"{situation}\n")

        except Exception as e:
            print(f"Error extracting situations: {e}")

    def run(self):
        """
        Run the full scraping process.
        """
        self.navigate_to_page("https://library.bu.edu/netadvantage")
        self.login()
        self.handle_passkey_prompt()
        self.click_trust_device()
        self.search_ticker()
        self.click_key_developments()
        self.click_expand_icon()
        self.extract_all_situations()

# Usage
scraper = SeleniumScraper(
    username="atharvam",
    password="QpAlZmwoskxn@00001",
    system_password="123456",  # Replace with your actual macOS system password
    ticker_symbol="AAPL"
)
scraper.run()
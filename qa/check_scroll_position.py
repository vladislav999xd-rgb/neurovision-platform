"""Проверить, что нажатие нижней кнопки не возвращает страницу наверх."""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


URL = "http://127.0.0.1:8501"

options = Options()
options.add_argument("--headless=new")
options.add_argument("--window-size=1536,900")
options.add_argument("--force-device-scale-factor=1")

driver = webdriver.Edge(options=options)
try:
    driver.get(URL)
    wait = WebDriverWait(driver, 30)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".medical-hero")))
    buttons = wait.until(
        lambda current: (
            found
            if len(found := current.find_elements(
                By.CSS_SELECTOR,
                '[data-testid="stButton"] button',
            )) >= 2
            else False
        )
    )
    joke_button = buttons[-2]
    old_joke = driver.find_element(By.CSS_SELECTOR, ".joke-text").text
    driver.execute_script(
        "arguments[0].scrollIntoView({block: 'center'})",
        joke_button,
    )
    before = driver.execute_script(
        "const main=document.querySelector('[data-testid=stMain]');"
        "return main ? main.scrollTop : window.scrollY;"
    )
    joke_button.click()
    wait.until(
        lambda current: current.find_element(By.CSS_SELECTOR, ".joke-text").text
        != old_joke
    )
    after = driver.execute_script(
        "const main=document.querySelector('[data-testid=stMain]');"
        "return main ? main.scrollTop : window.scrollY;"
    )
    difference = abs(after - before)
    assert difference < 80, (before, after, difference)
    print("Позиция до нажатия:", before)
    print("Позиция после нажатия:", after)
    print("Скачка страницы наверх нет: пройдено")
finally:
    driver.quit()

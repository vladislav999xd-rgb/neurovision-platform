"""Сохранить контрольный снимок современного интерфейса."""

from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "qa" / "registratura_modern_preview.png"
LOG = ROOT / "qa" / "streamlit_preview.log"
URL = "http://127.0.0.1:8522"


def wait_for_server(timeout=40):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(URL, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("Streamlit не запустился за отведённое время")


def main():
    log_handle = LOG.open("w", encoding="utf-8")
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless=true",
            "--server.address=127.0.0.1",
            "--server.port=8522",
            "--browser.gatherUsageStats=false",
            "--client.toolbarMode=minimal",
        ],
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    driver = None
    try:
        wait_for_server()
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1536,1500")
        options.add_argument("--force-device-scale-factor=1")
        options.add_argument("--disable-notifications")
        options.add_argument("--no-first-run")
        driver = webdriver.Edge(options=options)
        driver.get(URL)
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".medical-hero"))
        )
        time.sleep(3)
        page_height = driver.execute_script(
            "return Math.min(document.body.scrollHeight, 3600)"
        )
        driver.set_window_size(1536, page_height)
        time.sleep(1)
        driver.save_screenshot(str(OUTPUT))
        print(OUTPUT)
    finally:
        if driver is not None:
            driver.quit()
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        log_handle.close()


if __name__ == "__main__":
    main()

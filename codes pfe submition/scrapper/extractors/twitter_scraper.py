import httpx
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

async def get_twitter_metadata(url: str):
    """
    Extract text, images, and metadata from an X (Twitter) post.
    Uses Selenium because normal scraping cannot access protected media URLs.
    """

    # --------------------------
    # 1️⃣ FIRST TRY OEMBED (SAFE)
    # --------------------------
    try:
        async with httpx.AsyncClient() as client:
            api = f"https://publish.twitter.com/oembed?url={url}"
            r = await client.get(api, timeout=10)

            if r.status_code == 200:
                data = r.json()
                html = data.get("html", "")
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(" ").strip()

                return {
                    "text": text,
                    "images": [],
                    "metadata": data
                }
    except:
        pass

    # ---------------------------------------------------
    # 2️⃣ SELENIUM – LOAD FULL PAGE & EXTRACT EVERYTHING
    # ---------------------------------------------------
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(4)  # Wait for page + JS to load

        # ----------------------------
        # EXTRACT TEXT
        # ----------------------------
        text = ""
        try:
            # tweet text container
            element = driver.find_element(By.CSS_SELECTOR, "div[data-testid='tweetText']")
            text = element.text
        except:
            pass

        # ----------------------------
        # EXTRACT IMAGES (REAL LINKS)
        # ----------------------------
        images = []
        try:
            img_elements = driver.find_elements(By.CSS_SELECTOR, "img[src*='pbs.twimg.com']")
            for img in img_elements:
                src = img.get_attribute("src")
                if src not in images:
                    images.append(src)
        except:
            pass

        # ----------------------------
        # BUILD METADATA
        # ----------------------------
        metadata = {
            "image_count": len(images),
            "text_length": len(text),
            "url": url
        }

        driver.quit()

        return {
            "text": text,
            "images": images,
            "metadata": metadata
        }

    except Exception as e:
        return {
            "text": "",
            "images": [],
            "metadata": {"error": str(e)}
        }

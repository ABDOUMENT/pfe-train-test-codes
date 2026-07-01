import httpx
from bs4 import BeautifulSoup
import json
import re

async def scrape_reddit(url: str):
    # force old reddit (easier to parse)
    if "old.reddit.com" not in url:
        url = url.replace("www.reddit.com", "old.reddit.com")

    async with httpx.AsyncClient(follow_redirects=True) as client:
        r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})

    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}"}

    soup = BeautifulSoup(r.text, "html.parser")

    # -------- TEXT CONTENT --------
    text = ""
    txt_block = soup.find("div", class_="md")
    if txt_block:
        text = txt_block.get_text("\n").strip()

    # -------- IMAGES --------
    images = []

    # normal preview images
    img_tags = soup.find_all("img")
    for t in img_tags:
        src = t.get("src")
        if src and "preview" in src:
            images.append(src)

    # gallery JSON inside script tag
    scripts = soup.find_all("script")
    for s in scripts:
        if "window.___r" in s.text:
            try:
                json_str = re.search(r"window\.___r = (.*});", s.text).group(1)
                data = json.loads(json_str)
                media = data.get("posts", {}).get("models", {})
                for k, v in media.items():
                    if "media" in v and v["media"]:
                        m = v["media"]
                        if "content" in m:
                            images.append(m["content"])
            except:
                pass

    images = list(set(images))

    # -------- METADATA --------
    metadata = {
        "title": soup.title.string if soup.title else "",
        "url": url,
        "text_length": len(text),
        "images_count": len(images),
    }

    return {
        "text": text,
        "images": images,
        "metadata": metadata
    }

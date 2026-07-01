# extractors/article_scraper.py
import asyncio
import requests
from bs4 import BeautifulSoup

# optional: newspaper (blocking) -> call in thread
def _extract_with_newspaper(url: str):
    try:
        from newspaper import Article
    except Exception:
        return None

    a = Article(url)
    a.download()
    a.parse()
    return {
        "type": "article",
        "title": a.title,
        "text": a.text,
        "top_image": a.top_image,
        "authors": a.authors,
        "publish_date": a.publish_date.isoformat() if a.publish_date else None,
        "url": url,
    }

def _extract_with_bs(url: str):
    html = requests.get(url, timeout=15).text
    soup = BeautifulSoup(html, "html.parser")

    # --------------------------
    # EXTRACT ARTICLE TEXT
    # --------------------------
    article = soup.find("article")
    if article:
        paragraphs = article.find_all("p")
    else:
        paragraphs = soup.find_all("p")

    text = "\n\n".join(
        [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
    )

    # --------------------------
    # EXTRACT ALL IMAGES
    # --------------------------
    images = set()

    # 1) All <img> tags
    for img in soup.find_all("img"):
        src = img.get("src")
        if src and src.startswith("http"):
            images.add(src)

    # 2) OG images
    for tag in soup.find_all("meta", {"property": "og:image"}):
        if tag.get("content"):
            images.add(tag.get("content"))

    # 3) Twitter image tags
    for tag in soup.find_all("meta", {"name": "twitter:image"}):
        if tag.get("content"):
            images.add(tag.get("content"))

    images = list(images)

    # --------------------------
    # TITLE
    # --------------------------
    title = soup.title.string if soup.title else None

    # --------------------------
    # METADATA
    # --------------------------
    def meta(name, attr="property"):
        tag = soup.find("meta", {attr: name})
        return tag["content"] if tag and tag.get("content") else None

    author = meta("article:author") or meta("og:site_name")
    publish_date = meta("article:published_time")

    return {
        "type": "article",
        "title": title,
        "text": text,
        "images": images,
        "authors": [author] if author else [],
        "publish_date": publish_date,
        "url": url,
    }


async def extract_article_metadata(url: str):
    # try newspaper in thread
    try:
        res = await asyncio.to_thread(_extract_with_newspaper, url)
        if res and res.get("text"):
            return res
    except Exception:
        pass

    # fallback to bs extraction
    try:
        return await asyncio.to_thread(_extract_with_bs, url)
    except Exception as e:
        return {"error": str(e)}

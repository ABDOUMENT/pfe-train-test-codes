from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os
import json
import datetime
from pathlib import Path
import aiofiles
import httpx

from extractors import (
    get_instagram_metadata,
    get_twitter_metadata,
    extract_article_metadata,  # MUST be async now
    scrape_reddit
)

app = FastAPI()

DOWNLOAD_DIR = Path("download")
DOWNLOAD_DIR.mkdir(exist_ok=True)


# ============================================================
# Save content + metadata + images
# ============================================================
async def save_files(base_dir: Path, content: str, metadata: dict, images: list):
    base_dir.mkdir(parents=True, exist_ok=True)

    # Text content
    async with aiofiles.open(base_dir / "content.txt", "w", encoding="utf-8") as f:
        await f.write(content or "")

    # Metadata JSON
    async with aiofiles.open(base_dir / "metadata.json", "w", encoding="utf-8") as f:
        await f.write(json.dumps(metadata, ensure_ascii=False, indent=2))

    # Images
    async with httpx.AsyncClient() as client:
        for idx, img_url in enumerate(images):
            try:
                resp = await client.get(img_url, follow_redirects=True)
                if resp.status_code == 200:
                    ext = img_url.split("?")[0].split(".")[-1]
                    async with aiofiles.open(base_dir / f"image_{idx}.{ext}", "wb") as f:
                        await f.write(resp.content)
            except:
                continue


# ============================================================
# Routes
# ============================================================
@app.get("/")
def home():
    return {"status": "Metadata Extractor API running"}


@app.get("/extract")
async def extract(url: str):

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = DOWNLOAD_DIR / timestamp

    # ========================================================
    # INSTAGRAM
    # ========================================================
    if "instagram.com" in url:
        data = await get_instagram_metadata(url)
        await save_files(base_dir, data.get("text"), data, data.get("images", []))

        return {
            "content": str(base_dir / "content.txt"),
            "metadata": str(base_dir / "metadata.json"),
            "images": [str(base_dir / f) for f in os.listdir(base_dir) if f.startswith("image_")]
        }

    # ========================================================
    # TWITTER / X
    # ========================================================
    if "x.com" in url or "twitter.com" in url:
        data = await get_twitter_metadata(url)
        await save_files(base_dir, data.get("text"), data, data.get("images", []))

        return {
            "content": str(base_dir / "content.txt"),
            "metadata": str(base_dir / "metadata.json"),
            "images": [str(base_dir / f) for f in os.listdir(base_dir) if f.startswith("image_")]
        }

    # ========================================================
    # REDDIT
    # ========================================================
    if "reddit.com" in url:
        data = await scrape_reddit(url)
        await save_files(base_dir, data.get("text"), data, data.get("images", []))

        return {
            "content": str(base_dir / "content.txt"),
            "metadata": str(base_dir / "metadata.json"),
            "images": [str(base_dir / f) for f in os.listdir(base_dir) if f.startswith("image_")]
        }

    # ========================================================
    # ARTICLES (arXiv, blogs, newspapers, etc.)
    # ========================================================
    data = await extract_article_metadata(url)  # FIXED (async)
    await save_files(base_dir, data.get("text"), data, data.get("images", []))

    return {
        "content": str(base_dir / "content.txt"),
        "metadata": str(base_dir / "metadata.json"),
        "images": [str(base_dir / f) for f in os.listdir(base_dir) if f.startswith("image_")]
    }

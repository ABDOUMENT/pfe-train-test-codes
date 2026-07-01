import instaloader
from urllib.parse import urlparse
import asyncio


def _sync_instagram_scrape(url: str):
    """Run Instaloader synchronously inside a thread-safe wrapper."""
    loader = instaloader.Instaloader(
        download_videos=False,
        download_video_thumbnails=False,
        download_comments=False,
        save_metadata=False,
        quiet=True
    )

    # Extract shortcode from URL
    parsed = urlparse(url)
    shortcode = parsed.path.split("/")[-2]

    post = instaloader.Post.from_shortcode(loader.context, shortcode)

    # Collect images
    images = []
    if post.typename == "GraphSidecar":
        for node in post.get_sidecar_nodes():
            images.append(node.display_url)
    else:
        images.append(post.url)

    # Metadata (full)
    metadata = {
        "shortcode": post.shortcode,
        "post_url": f"https://www.instagram.com/p/{post.shortcode}/",
        "owner_username": post.owner_username,
        "caption": post.caption or "",
        "hashtags": list(post.caption_hashtags),
        "mentions": list(post.caption_mentions),
        "date_utc": str(post.date_utc),
        "likes": post.likes,
        "is_video": post.is_video,
        "video_url": post.video_url if post.is_video else None,
        "images": images,
    }

    # TEXT CONTENT (main caption)
    text = post.caption or ""

    return {
        "text": text,
        "images": images,
        "metadata": metadata
    }


async def get_instagram_metadata(url: str):
    """Async wrapper to make Instaloader non-blocking."""
    return await asyncio.to_thread(_sync_instagram_scrape, url)

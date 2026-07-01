def detect_platform(url: str):
    url = url.lower()

    if "instagram.com" in url:
        return "instagram"

    if "x.com" in url or "twitter.com" in url:
        return "twitter"

    return "article"

import os
import time
import random
import logging
import hashlib
import requests
from pathlib import Path
from datetime import datetime

SPECIES = [
    "Hibou grand-duc",
    "Flamant rose",
    "Martin-pêcheur",
    "Cygne tuberculé",
    "Pic vert",
]

SPECIES_EN = {
    "Hibou grand-duc": "Eurasian eagle-owl bird",
    "Flamant rose": "Greater flamingo bird",
    "Martin-pêcheur": "Common kingfisher bird",
    "Cygne tuberculé": "Mute swan bird",
    "Pic vert": "European green woodpecker bird",
}

BASE_DIR = Path("../data/raw")
LOG_DIR = Path("../logs")

IMAGES_PER_SPECIES = 100
REQUEST_DELAY = (1.0, 2.5)
TIMEOUT = 15
MAX_RETRIES = 3

FLICKR_API = "https://api.flickr.com/services/feeds/photos_public.gne"
INAT_API = "https://api.inaturalist.org/v1/observations"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"scraping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def slug(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("â", "a")
        .replace("û", "u")
        .replace("î", "i")
        .lower()
    )


def random_delay():
    time.sleep(random.uniform(*REQUEST_DELAY))


# for diplucatess
def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_flickr_urls(query: str, count: int = 130) -> list[str]:

    urls: list[str] = []
    session = requests.Session()
    session.headers.update(HEADERS)

    tags_base = ",".join(query.split())
    tag_variants = [
        tags_base,
        tags_base + ",nature",
        tags_base + ",wildlife",
        tags_base + ",ornithology",
        tags_base + ",photography",
    ]

    for tag_set in tag_variants:
        if len(urls) >= count:
            break
        params = {
            "format": "json",
            "nojsoncallback": "1",
            "tags": tag_set,
            "tagmode": "all",
            "lang": "en-us",
        }
        try:
            random_delay()
            resp = session.get(FLICKR_API, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Flickr error (tags='{tag_set}'): {e}")
            continue

        items = data.get("items", [])
        logger.info(f"  Flickr tags='{tag_set}': {len(items)} results")

        for item in items:
            media_url = item.get("media", {}).get("m", "")
            if media_url:
                # Replace '_m' (small) with '_b' (large) in the URL
                large_url = media_url.replace("_m.jpg", "_b.jpg")
                if large_url not in urls:
                    urls.append(large_url)

    return urls


def fetch_inaturalist_urls(query: str, count: int = 100) -> list[str]:
    urls: list[str] = []
    session = requests.Session()
    session.headers.update(HEADERS)

    # fetch multiple pages to hit the count target
    per_page = min(count, 200)
    for page in range(1, 4):
        if len(urls) >= count:
            break
        params = {
            "q": query,
            "iconic_taxa": "Aves",  # birds only — key filter
            "has[]": "photos",
            "quality_grade": "research",  # community-verified IDs
            "per_page": per_page,
            "page": page,
        }
        try:
            random_delay()
            resp = session.get(INAT_API, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"iNaturalist error (page {page}): {e}")
            break

        results = data.get("results", [])
        logger.info(f"  iNaturalist page {page}: {len(results)} observations")

        for obs in results:
            for photo in obs.get("photos", []):
                url = photo.get("url", "").replace("square", "large")
                if url and url not in urls:
                    urls.append(url)

        if len(results) < per_page:
            break  # no more pages

    return urls[:count]


def fetch_image_urls(species: str, count: int) -> list[str]:
    query = SPECIES_EN[species]
    urls: list[str] = []

    logger.info(f"  Collecting from iNaturalist…")
    urls += fetch_inaturalist_urls(query, count)

    if len(urls) < count:
        logger.info(f"  Supplementing with Flickr ({count - len(urls)} needed)…")
        flickr = fetch_flickr_urls(query, count - len(urls) + 20)
        for u in flickr:
            if u not in urls:
                urls.append(u)

    logger.info(f"  Total unique URLs: {len(urls)}")
    return urls[: count + 20]  # small buffer for failures


def download_image(url: str, dest_path: Path, retries: int = MAX_RETRIES) -> bool:
    for attempt in range(1, retries + 1):
        try:
            random_delay()
            resp = requests.get(url, timeout=TIMEOUT, headers=HEADERS, stream=True)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "image" not in content_type:
                logger.warning(f"Non-image content ({content_type}): {url}")
                return False

            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            # smaller than 1kb
            if dest_path.stat().st_size < 1024:
                logger.warning(f"File too small, skipping: {url}")
                dest_path.unlink(missing_ok=True)
                return False

            return True

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout (attempt {attempt}/{retries}): {url}")
        except requests.exceptions.HTTPError as e:
            logger.warning(
                f"HTTP {e.response.status_code} (attempt {attempt}/{retries}): {url}"
            )
            if e.response.status_code in (403, 404, 410):
                break
        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error (attempt {attempt}/{retries}): {e}")
        except OSError as e:
            logger.warning(f"Disk error: {e}")
            break

        if attempt < retries:
            time.sleep(2**attempt)

    dest_path.unlink(missing_ok=True)
    return False


def scrape_species(species: str):
    folder = BASE_DIR / slug(species)
    folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'═'*60}")
    logger.info(f"Species : {species}  →  '{SPECIES_EN[species]}'")
    logger.info(f"Folder  : {folder}")
    logger.info(f"{'═'*60}")

    urls = fetch_image_urls(species, IMAGES_PER_SPECIES)

    downloaded = 0
    failed = 0
    seen_hashes: set[str] = set()

    for idx, url in enumerate(urls):
        if downloaded >= IMAGES_PER_SPECIES:
            break

        ext = url.split("?")[0].split(".")[-1].lower()
        if ext not in ("jpg", "jpeg", "png", "webp"):
            ext = "jpg"

        filename = folder / f"{slug(species)}_{idx + 1:04d}.{ext}"
        logger.info(f"  [{idx + 1}/{len(urls)}] → {filename.name}")

        if download_image(url, filename):
            file_hash = sha256_of_file(filename)
            if file_hash in seen_hashes:
                logger.info(f"  Duplicate, removed: {filename.name}")
                filename.unlink()
            else:
                seen_hashes.add(file_hash)
                downloaded += 1
                logger.info(f"  ✓ {downloaded}/{IMAGES_PER_SPECIES}: {filename.name}")
        else:
            failed += 1
            logger.error(f"  ✗ Failed: {url}")

    logger.info(f"\n  Summary '{species}': {downloaded} downloaded, {failed} failed.")


def main():
    logger.info("Scraping started")
    logger.info(f"Log file: {log_file}\n")

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    for species in SPECIES:
        scrape_species(species)

    logger.info("\nAll species done.")


if __name__ == "__main__":
    main()

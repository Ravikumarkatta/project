import argparse
import logging
import sys
import os
import requests
import json
import zipfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("cli")

DOWNLOAD_DIR = "downloads"
BIBLE_SOURCES_FILE = "config/bible_sources.json"

def ensure_download_dir():
    """
    Ensures the download directory exists.
    """
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        logger.info("Created download directory: %s", DOWNLOAD_DIR)

def load_bible_sources():
    """
    Loads the Bible sources from the JSON file.
    :return: A list of Bible sources.
    """
    if not os.path.exists(BIBLE_SOURCES_FILE):
        logger.error("Bible sources file not found: %s", BIBLE_SOURCES_FILE)
        sys.exit(1)

    with open(BIBLE_SOURCES_FILE, 'r') as f:
        try:
            sources = json.load(f)
            return sources.get("translations", [])
        except json.JSONDecodeError as e:
            logger.error("Failed to parse Bible sources file: %s", e)
            sys.exit(1)

def extract_usfx(zip_path, destination_dir):
    """
    Extracts USFX files from a ZIP archive.
    :param zip_path: Path to the ZIP file.
    :param destination_dir: Directory to extract the USFX files to.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination_dir)
        logger.info("Extracted USFX files from %s to %s", zip_path, destination_dir)
    except zipfile.BadZipFile as e:
        logger.error("Failed to extract ZIP file %s: %s", zip_path, e)
        raise

def download_file(url, destination):
    """
    Downloads a file from a URL and saves it to the destination.
    If the file is a ZIP, it extracts the contents.
    :param url: The URL of the file to download.
    :param destination: The local file path to save the downloaded file.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Downloaded file from %s to %s", url, destination)

        # Check if the file is a ZIP and extract it
        if destination.endswith('.zip'):
            extract_usfx(destination, DOWNLOAD_DIR)
            os.remove(destination)  # Remove the ZIP file after extraction

    except requests.RequestException as e:
        logger.error("Failed to download file from %s: %s", url, e)
        raise

def download_bible(version):
    """
    Downloads a specific Bible translation.
    :param version: The ID of the Bible version to download.
    """
    sources = load_bible_sources()
    bible = next((b for b in sources if b["id"].lower() == version.lower()), None)

    if not bible:
        logger.error("Bible version '%s' not found in sources.", version)
        return

    file_path = os.path.join(DOWNLOAD_DIR, f"{bible['id']}.zip" if bible["format"] == "usfx" else f"{bible['id']}.txt")
    try:
        logger.info("Downloading '%s'...", bible["name"])
        download_file(bible["url"], file_path)
        logger.info("Successfully downloaded '%s' to %s.", bible["name"], file_path)
    except Exception as e:
        logger.error("Failed to download '%s': %s", bible["name"], e)

def download_all_bibles():
    """
    Downloads all available Bible translations.
    """
    sources = load_bible_sources()
    ensure_download_dir()

    for bible in sources:
        logger.info("Downloading '%s'...", bible["name"])
        try:
            download_bible(bible["id"])
        except Exception as e:
            logger.error("Failed to download '%s': %s", bible['name'], e)

    logger.info("All Bible translations have been downloaded.")

def main():
    try:
        parser = argparse.ArgumentParser(description="Bible CLI tool")
        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Subparser for the "download-bible" command
        parser_download = subparsers.add_parser('download-bible', help='Download a specific Bible translation')
        parser_download.add_argument('--version', required=False, help='ID of the Bible version to download (e.g., KJV)')

        # Subparser for the "download-all" command
        subparsers.add_parser('download-all', help='Download all available Bible translations')

        args = parser.parse_args()

        if args.command == 'download-bible':
            if not args.version:
                logger.error("Please specify a Bible version using --version.")
                sys.exit(1)
            download_bible(args.version)
        elif args.command == 'download-all':
            download_all_bibles()
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        logger.exception("An error occurred: %s", e)
        sys.exit(1)

if __name__ == '__main__':
    main()
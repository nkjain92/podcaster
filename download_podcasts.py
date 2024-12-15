import xml.etree.ElementTree as ET
import aiohttp
import asyncio
import os
import json
from pathlib import Path
from tqdm import tqdm
import re
from typing import Dict, List
import logging
import ssl
import certifi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('podcast_downloader.log'),
        logging.StreamHandler()
    ]
)

# Constants
XML_PATH = '/Users/nishankjain/projects/podcaster/podcast_links.rss'
PODCASTS_DIR = '/Users/nishankjain/projects/podcaster/podcasts'
TRACKING_FILE = 'download_tracking.json'
CONCURRENT_DOWNLOADS = 5

class PodcastDownloader:
    def __init__(self):
        # Delete existing tracking file if it exists
        if os.path.exists(TRACKING_FILE):
            os.remove(TRACKING_FILE)
        self.download_tracking = self._load_tracking()
        self.ensure_directories()

    def ensure_directories(self):
        """Ensure necessary directories exist"""
        Path(PODCASTS_DIR).mkdir(parents=True, exist_ok=True)

    def _load_tracking(self) -> Dict:
        """Load download tracking information"""
        if os.path.exists(TRACKING_FILE):
            with open(TRACKING_FILE, 'r') as f:
                return json.load(f)
        return {'downloaded': [], 'failed': []}

    def _save_tracking(self):
        """Save download tracking information"""
        with open(TRACKING_FILE, 'w') as f:
            json.dump(self.download_tracking, f, indent=2)

    def sanitize_filename(self, filename: str) -> str:
        """Remove special characters from filename"""
        # First replace slashes with dashes
        filename = filename.replace('/', '-').replace('\\', '-')
        # Then remove other invalid characters
        filename = re.sub(r'[<>:"|?*]', '', filename)
        # Limit filename length
        return filename[:200]

    def parse_xml(self) -> List[Dict]:
        """Parse XML file and extract episode information"""
        try:
            # Read the file and find where the actual XML starts
            with open(XML_PATH, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find where the RSS tag starts
            xml_start = content.find('<rss')
            if xml_start == -1:
                logging.error("Could not find RSS tag in file")
                return []

            # Create a clean XML string starting from the RSS tag
            clean_xml = content[xml_start:]

            # Clean up problematic characters
            # Replace smart quotes and other problematic characters
            replacements = {
                '"': '"',  # Smart quotes
                '"': '"',
                ''': "'",
                ''': "'",
                '–': '-',  # En dash
                '—': '-',  # Em dash
                '…': '...',
                '\x0b': '',  # Vertical tab
                '\x0c': '',  # Form feed
                '&': '&amp;',  # Ensure proper XML escaping
                '<![CDATA[': '',  # Remove CDATA tags
                ']]>': ''
            }

            for old, new in replacements.items():
                clean_xml = clean_xml.replace(old, new)

            # Debug: Write the cleaned XML to a file for inspection
            with open('cleaned_podcast.xml', 'w', encoding='utf-8') as f:
                f.write(clean_xml)

            try:
                # Try to parse the cleaned XML
                root = ET.fromstring(clean_xml)
            except ET.ParseError as pe:
                # If parsing fails, print the problematic line
                lines = clean_xml.split('\n')
                error_line = int(str(pe).split('line')[1].split(',')[0].strip())
                context_start = max(0, error_line - 2)
                context_end = min(len(lines), error_line + 2)

                logging.error(f"Parse error near these lines:")
                for i in range(context_start, context_end):
                    logging.error(f"Line {i+1}: {lines[i]}")
                raise

            # Find all item elements under channel
            channel = root.find('channel')
            if channel is None:
                logging.error("No channel element found in RSS")
                return []

            items = channel.findall('item')
            logging.info(f"Found {len(items)} items in RSS feed")

            episodes = []
            for item in items:
                title_elem = item.find('title')
                enclosure = item.find('enclosure')

                if title_elem is not None and enclosure is not None:
                    title = title_elem.text
                    url = enclosure.get('url')
                    logging.info(f"Found episode: {title}")
                    episodes.append({
                        'title': title,
                        'url': url
                    })
                else:
                    logging.warning(f"Missing title or enclosure in item {len(episodes) + 1}")

            logging.info(f"Total episodes parsed: {len(episodes)}")
            return episodes

        except Exception as e:
            logging.error(f"Error parsing XML: {str(e)}")
            raise

    async def download_episode(self, session: aiohttp.ClientSession, episode: Dict, pbar: tqdm) -> bool:
        """Download a single episode"""
        filename = self.sanitize_filename(episode['title']) + '.mp3'
        filepath = os.path.join(PODCASTS_DIR, filename)

        if os.path.exists(filepath):
            logging.info(f"Skipping {filename} - already exists")
            return True

        try:
            async with session.get(episode['url']) as response:
                if response.status != 200:
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status
                    )

                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                with open(filepath, 'wb') as f:
                    downloaded = 0
                    async for data in response.content.iter_chunked(block_size):
                        f.write(data)
                        downloaded += len(data)
                        pbar.update(len(data))

            return True

        except Exception as e:
            logging.error(f"Failed to download {episode['title']}: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return False

    async def download_all(self):
        """Download all episodes"""
        episodes = self.parse_xml()
        episodes_to_download = [
            ep for ep in episodes
            if ep['title'] not in self.download_tracking['downloaded']
        ]

        logging.info(f"Total episodes found: {len(episodes)}")
        logging.info(f"Episodes to download: {len(episodes_to_download)}")

        if not episodes_to_download:
            logging.info("No new episodes to download")
            return

        logging.info(f"Starting download of {len(episodes_to_download)} episodes")

        # Create SSL context with certifi certificates
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        # Create ClientSession with the SSL context
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            with tqdm(total=len(episodes_to_download), desc="Overall Progress") as pbar:
                for i in range(0, len(episodes_to_download), CONCURRENT_DOWNLOADS):
                    batch = episodes_to_download[i:i + CONCURRENT_DOWNLOADS]
                    tasks = []
                    for episode in batch:
                        task = asyncio.create_task(
                            self.download_episode(session, episode, pbar)
                        )
                        tasks.append((episode, task))

                    for episode, task in tasks:
                        success = await task
                        if success:
                            self.download_tracking['downloaded'].append(episode['title'])
                            if episode['title'] in self.download_tracking['failed']:
                                self.download_tracking['failed'].remove(episode['title'])
                        else:
                            self.download_tracking['failed'].append(episode['title'])

                    self._save_tracking()

async def main():
    downloader = PodcastDownloader()
    await downloader.download_all()

if __name__ == "__main__":
    asyncio.run(main())
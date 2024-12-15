import os
import json
import time
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
import ssl
import certifi
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from deepgram import Deepgram
except ImportError:
    print("Deepgram SDK not found. Installing specific version...")
    os.system('pip install deepgram-sdk==2.12.0')
    from deepgram import Deepgram

# Configure logging
logging.basicConfig(
    filename='transcription.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
PODCASTS_DIR = "/Users/nishankjain/projects/podcaster/podcasts"
TRANSCRIPTIONS_DIR = "transcriptions"
MIME_TYPE = "audio/mp3"
MAX_RETRIES = 3
MAX_FILE_SIZE = 300 * 1024 * 1024  # 300MB limit

# Set SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())

def get_file_size_mb(file_path):
    """Return file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

async def transcribe_audio(filepath, dg_client):
    """Transcribe a single audio file using Deepgram"""
    file_size = get_file_size_mb(filepath)
    retries = 0

    # Check file size
    if file_size > MAX_FILE_SIZE:
        logging.error(f"File {filepath} is too large ({file_size:.2f} MB). Skipping.")
        return None

    while retries < MAX_RETRIES:
        try:
            with open(filepath, 'rb') as audio:
                source = {'buffer': audio, 'mimetype': MIME_TYPE}
                response = await dg_client.transcription.prerecorded(
                    source,
                    {
                        'smart_format': True,
                        'punctuate': True,
                        'tier': 'base',
                    },
                    timeout=300,  # 5 minutes timeout
                    ssl=ssl_context
                )

                # Check if response is valid
                if not response or 'results' not in response:
                    logging.error(f"Invalid response from Deepgram for {filepath}: {response}")
                    raise Exception("Invalid response format")

                return response

        except Exception as e:
            retries += 1
            error_msg = str(e)
            logging.error(f"Attempt {retries} failed for {filepath}: {error_msg}")

            # Check for specific error types
            if "Invalid API key" in error_msg:
                logging.error("Invalid API key. Please check your Deepgram API key.")
                return None
            elif "timeout" in error_msg.lower():
                logging.error(f"Timeout error for {filepath}. File might be too large.")
                await asyncio.sleep(30)  # Longer wait for timeout errors
            elif "rate limit" in error_msg.lower():
                wait_time = min(30 * (2 ** retries), 300)  # Exponential backoff, max 5 minutes
                logging.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry.")
                await asyncio.sleep(wait_time)
            else:
                wait_time = min(10 * (2 ** retries), 120)  # Exponential backoff, max 2 minutes
                logging.error(f"Unknown error. Waiting {wait_time} seconds before retry.")
                await asyncio.sleep(wait_time)

            if retries == MAX_RETRIES:
                logging.error(f"All attempts failed for {filepath}. Last error: {error_msg}")
                return None

async def main():
    # Initialize Deepgram
    try:
        dg_client = Deepgram(DEEPGRAM_API_KEY)
    except Exception as e:
        logging.error(f"Failed to initialize Deepgram client: {str(e)}")
        print("Failed to initialize Deepgram client. Check your API key.")
        return

    # Ensure transcriptions directory exists
    os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)

    # Get list of MP3 files and sort by size (smallest first)
    mp3_files = []
    for f in os.listdir(PODCASTS_DIR):
        if f.endswith('.mp3'):
            file_path = os.path.join(PODCASTS_DIR, f)
            size = get_file_size_mb(file_path)
            mp3_files.append((f, size))

    mp3_files.sort(key=lambda x: x[1])  # Sort by size
    mp3_files = [f[0] for f in mp3_files]  # Get just the filenames
    total_files = len(mp3_files)

    # Load progress from previous run if exists
    progress_file = "transcription_progress.json"
    completed_files = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            completed_files = set(json.load(f))

    logging.info(f"Starting transcription of {total_files} files")
    print(f"Found {total_files} files to process")
    print(f"Already completed: {len(completed_files)} files")

    # Track progress
    completed = len(completed_files)
    failed = 0

    try:
        for mp3_file in tqdm(mp3_files, desc="Transcribing files"):
            if mp3_file in completed_files:
                continue

            input_path = os.path.join(PODCASTS_DIR, mp3_file)
            output_path = os.path.join(TRANSCRIPTIONS_DIR, mp3_file.replace('.mp3', '.json'))

            file_size = get_file_size_mb(input_path)
            logging.info(f"Processing: {mp3_file} (Size: {file_size:.2f} MB)")
            print(f"\nCurrently processing: {mp3_file} (Size: {file_size:.2f} MB)")

            try:
                # Transcribe file
                response = await transcribe_audio(input_path, dg_client)

                if response:
                    # Save transcription
                    with open(output_path, 'w') as f:
                        json.dump(response, f, indent=2)
                    completed += 1
                    completed_files.add(mp3_file)
                    # Save progress
                    with open(progress_file, 'w') as f:
                        json.dump(list(completed_files), f)
                    logging.info(f"Successfully transcribed: {mp3_file}")
                else:
                    failed += 1
                    logging.error(f"Failed to transcribe: {mp3_file}")

                # Add a longer delay between files
                await asyncio.sleep(5)

            except Exception as e:
                failed += 1
                logging.error(f"Error processing {mp3_file}: {str(e)}")
                await asyncio.sleep(10)  # Longer delay after errors

    except KeyboardInterrupt:
        print("\nScript interrupted by user. Progress has been saved.")
        logging.info("Script interrupted by user")
    finally:
        # Log final statistics
        logging.info(f"Transcription session ended. Successfully transcribed: {completed}, Failed: {failed}")
        print(f"\nTranscription session ended!")
        print(f"Successfully transcribed: {completed}")
        print(f"Failed: {failed}")
        print(f"Progress saved to {progress_file}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

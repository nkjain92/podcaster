import os
import json
import logging
from pathlib import Path
from tqdm import tqdm
import re

# Configure logging
logging.basicConfig(
    filename='transcript_cleaning.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
TRANSCRIPTIONS_DIR = "transcriptions"
CLEANED_DIR = "cleaned_transcripts"
CLEANING_PROGRESS_FILE = "cleaning_progress.json"

# Patterns to remove (can be extended based on needs)
PATTERNS_TO_REMOVE = [
    r'\[music\]',
    r'\[applause\]',
    r'\[laughter\]',
    r'\[background noise\]',
    r'\[silence\]',
    r'\[inaudible\]',
    r'\[crosstalk\]',
    # Add more patterns as needed
]

def load_progress():
    """Load the progress of cleaned files"""
    if os.path.exists(CLEANING_PROGRESS_FILE):
        with open(CLEANING_PROGRESS_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_progress(cleaned_files):
    """Save the progress of cleaned files"""
    with open(CLEANING_PROGRESS_FILE, 'w') as f:
        json.dump(list(cleaned_files), f)

def clean_text(text):
    """Clean the transcript text by removing unwanted patterns"""
    cleaned = text
    for pattern in PATTERNS_TO_REMOVE:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def process_transcript(input_file):
    """Process a single transcript file"""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Extract the main transcript
        transcript = data.get('results', {}).get('channels', [{}])[0].get('alternatives', [{}])[0].get('transcript', '')

        if not transcript:
            raise ValueError("No transcript found in the JSON structure")

        # Clean the transcript
        cleaned_transcript = clean_text(transcript)

        # Create simplified JSON structure
        cleaned_data = {
            "transcript": cleaned_transcript
        }

        return cleaned_data

    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in {input_file}: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error processing {input_file}: {str(e)}")
        return None

def main():
    # Ensure output directory exists
    os.makedirs(CLEANED_DIR, exist_ok=True)

    # Load progress
    cleaned_files = load_progress()

    # Get list of all JSON files in transcriptions directory
    transcript_files = [f for f in os.listdir(TRANSCRIPTIONS_DIR) if f.endswith('.json')]
    total_files = len(transcript_files)

    logging.info(f"Starting cleaning process for {total_files} files")
    print(f"Found {total_files} files to process")
    print(f"Already cleaned: {len(cleaned_files)} files")

    # Track progress
    successfully_cleaned = 0
    failed = 0

    try:
        for filename in tqdm(transcript_files, desc="Cleaning transcripts"):
            if filename in cleaned_files:
                continue

            input_path = os.path.join(TRANSCRIPTIONS_DIR, filename)
            output_path = os.path.join(CLEANED_DIR, filename)

            logging.info(f"Processing: {filename}")
            print(f"\nCurrently processing: {filename}")

            try:
                cleaned_data = process_transcript(input_path)

                if cleaned_data:
                    # Save cleaned transcript
                    with open(output_path, 'w') as f:
                        json.dump(cleaned_data, f, indent=2)
                    successfully_cleaned += 1
                    cleaned_files.add(filename)
                    save_progress(cleaned_files)
                    logging.info(f"Successfully cleaned: {filename}")
                else:
                    failed += 1
                    logging.error(f"Failed to clean: {filename}")

            except Exception as e:
                failed += 1
                logging.error(f"Error processing {filename}: {str(e)}")

    except KeyboardInterrupt:
        print("\nScript interrupted by user. Progress has been saved.")
        logging.info("Script interrupted by user")

    finally:
        # Log final statistics
        logging.info(f"Cleaning process ended. Successfully cleaned: {successfully_cleaned}, Failed: {failed}")
        print(f"\nCleaning process ended!")
        print(f"Successfully cleaned: {successfully_cleaned}")
        print(f"Failed: {failed}")
        print(f"Progress saved to {CLEANING_PROGRESS_FILE}")

if __name__ == "__main__":
    main()
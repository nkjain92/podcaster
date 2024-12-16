import json
import sys
import os
from datetime import datetime
from src.extract.extract_founder import extract_founder_data

def save_founder_data(founder_data, transcript_filename):
    """Save founder data to a JSON file"""
    # Create output directory if it doesn't exist
    output_dir = "extracted_data"
    os.makedirs(output_dir, exist_ok=True)

    # Get founder name from basic info or use filename
    founder_name = founder_data.basic_info.name.replace(" ", "_").lower()
    if not founder_name:
        founder_name = os.path.splitext(os.path.basename(transcript_filename))[0]

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{output_dir}/{founder_name}_{timestamp}.json"

    # Save to file
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(founder_data.to_json(), f, indent=2, ensure_ascii=False)

    print(f"Saved founder data to: {output_filename}")

def main():
    try:
        # Load the transcript for Elon Musk
        with open("cleaned_transcripts/#1 Elon Musk Tesla, SpaceX, & the Quest for a Fantastic Future.json", "r") as file:
            transcript_data = json.load(file)
            transcript = transcript_data["transcript"]

        # Extract founder data
        founder_data = extract_founder_data(transcript)

        # Save the extracted data
        save_founder_data(founder_data, "#1 Elon Musk Tesla, SpaceX, & the Quest for a Fantastic Future.json")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
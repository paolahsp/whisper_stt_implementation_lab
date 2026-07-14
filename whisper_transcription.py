import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
CHUNKS_DIR = BASE_DIR / "chunks"
OUTPUTS_DIR = BASE_DIR / "outputs"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
REPORT_DIR = BASE_DIR / "report"


def create_project_directories() -> None:
    """
    Create the project directories if they do not already exist.
    """
    directories = [
        AUDIO_DIR,
        CHUNKS_DIR,
        OUTPUTS_DIR,
        SCREENSHOTS_DIR,
        REPORT_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("Project directories verified successfully.")


def load_openai_client() -> OpenAI:
    """
    Load the OpenAI API key from the .env file
    and initialize the OpenAI client.
    """
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found. "
            "Add it to the .env file before running the program."
        )

    if api_key == "your_openai_api_key_here":
        raise ValueError(
            "The placeholder API key is still configured. "
            "Replace it with your real OpenAI API key in the .env file."
        )

    client = OpenAI(api_key=api_key)

    print("OpenAI client initialized successfully.")

    return client


def main() -> None:
    """
    Run the initial project setup checks.
    """
    print("=" * 60)
    print("WHISPER STT IMPLEMENTATION LAB")
    print("=" * 60)

    create_project_directories()

    try:
        load_openai_client()
    except ValueError as error:
        print(f"Configuration error: {error}")

    print("=" * 60)
    print("Initial setup check completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
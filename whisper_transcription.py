import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment


# ---------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
CHUNKS_DIR = BASE_DIR / "chunks"
OUTPUTS_DIR = BASE_DIR / "outputs"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
REPORT_DIR = BASE_DIR / "report"

SUPPORTED_AUDIO_EXTENSIONS = [
    ".mp3",
    ".wav",
    ".m4a",
    ".mpeg",
    ".mp4",
    ".webm",
]

GUIDED_TRANSCRIPTION_PROMPT = (
    "This audio is an English conversation about the Apollo Moon mission, "
    "astronauts returning from the Moon, lunar exploration, spacecraft, "
    "rendezvous, orbiting the Moon, lunar rocks, soil samples, and future "
    "space missions. Preserve natural punctuation and use accurate space "
    "exploration terminology."
)


# ---------------------------------------------------------
# PROJECT SETUP
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# AUDIO FILE HANDLING
# ---------------------------------------------------------

def find_audio_file() -> Path:
    """
    Find the first supported audio file in the audio directory.
    """
    audio_files = [
        file_path
        for file_path in AUDIO_DIR.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]

    if not audio_files:
        raise FileNotFoundError(
            "No supported audio file was found in the audio directory. "
            "Add an audio file such as .mp3, .wav, or .m4a."
        )

    audio_files.sort()

    audio_file = audio_files[0]

    print(f"Audio file found: {audio_file.name}")

    return audio_file


def analyze_audio_file(audio_file_path: Path) -> AudioSegment:
    """
    Load the audio file and print its basic properties.
    """
    audio = AudioSegment.from_file(audio_file_path)

    duration_seconds = len(audio) / 1000
    duration_minutes = duration_seconds / 60
    file_size_mb = audio_file_path.stat().st_size / (1024 * 1024)

    print("Audio analysis completed successfully.")
    print(f"File name: {audio_file_path.name}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Duration: {duration_seconds:.2f} seconds")
    print(f"Duration: {duration_minutes:.2f} minutes")
    print(f"Channels: {audio.channels}")
    print(f"Frame rate: {audio.frame_rate} Hz")
    print(f"Sample width: {audio.sample_width} bytes")

    return audio


# ---------------------------------------------------------
# TRANSCRIPTION FUNCTIONS
# ---------------------------------------------------------

def transcribe_audio_unguided(
    client: OpenAI,
    audio_file_path: Path,
) -> str:
    """
    Transcribe an audio file without a prompt.
    """
    print("Starting unguided transcription...")

    with audio_file_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            language="en",
        )

    transcription_text = str(transcription).strip()

    if not transcription_text:
        raise ValueError(
            "The unguided transcription returned no text."
        )

    print("Unguided transcription completed successfully.")

    return transcription_text


def transcribe_audio_guided(
    client: OpenAI,
    audio_file_path: Path,
    prompt: str,
) -> str:
    """
    Transcribe an audio file using contextual guidance.
    """
    print("Starting guided transcription...")

    with audio_file_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            language="en",
            prompt=prompt,
        )

    transcription_text = str(transcription).strip()

    if not transcription_text:
        raise ValueError(
            "The guided transcription returned no text."
        )

    print("Guided transcription completed successfully.")

    return transcription_text


# ---------------------------------------------------------
# OUTPUT FILE HANDLING
# ---------------------------------------------------------

def save_text_transcription(
    transcription_text: str,
    output_file_name: str,
) -> Path:
    """
    Save transcription text to a plain text file.
    """
    output_path = OUTPUTS_DIR / output_file_name

    output_path.write_text(
        transcription_text + "\n",
        encoding="utf-8",
    )

    print(f"Transcription saved to: {output_path}")

    return output_path


def save_comparison_file(
    unguided_text: str,
    guided_text: str,
) -> Path:
    """
    Save both transcription results in one comparison file.
    """
    comparison_path = OUTPUTS_DIR / "transcription_comparison.txt"

    comparison_content = (
        "=" * 60
        + "\nUNGUIDED TRANSCRIPTION\n"
        + "=" * 60
        + "\n"
        + unguided_text
        + "\n\n"
        + "=" * 60
        + "\nGUIDED TRANSCRIPTION\n"
        + "=" * 60
        + "\n"
        + guided_text
        + "\n"
    )

    comparison_path.write_text(
        comparison_content,
        encoding="utf-8",
    )

    print(f"Comparison saved to: {comparison_path}")

    return comparison_path


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

def main() -> None:
    """
    Analyze the audio, create guided and unguided
    transcriptions, and save the results.
    """
    print("=" * 60)
    print("WHISPER STT IMPLEMENTATION LAB")
    print("=" * 60)

    create_project_directories()

    try:
        client = load_openai_client()

        audio_file = find_audio_file()

        analyze_audio_file(audio_file)

        print("-" * 60)
        print("UNGUIDED APPROACH")
        print("-" * 60)

        unguided_transcription = transcribe_audio_unguided(
            client=client,
            audio_file_path=audio_file,
        )

        print(unguided_transcription)

        save_text_transcription(
            transcription_text=unguided_transcription,
            output_file_name="unguided_transcription.txt",
        )

        print("-" * 60)
        print("GUIDED APPROACH")
        print("-" * 60)

        guided_transcription = transcribe_audio_guided(
            client=client,
            audio_file_path=audio_file,
            prompt=GUIDED_TRANSCRIPTION_PROMPT,
        )

        print(guided_transcription)

        save_text_transcription(
            transcription_text=guided_transcription,
            output_file_name="guided_transcription.txt",
        )

        save_comparison_file(
            unguided_text=unguided_transcription,
            guided_text=guided_transcription,
        )

    except FileNotFoundError as error:
        print(f"File error: {error}")

    except ValueError as error:
        print(f"Configuration error: {error}")

    except Exception as error:
        print(
            f"Unexpected error: "
            f"{type(error).__name__}: {error}"
        )

    print("=" * 60)
    print("Program completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
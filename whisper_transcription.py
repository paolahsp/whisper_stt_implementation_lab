import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

CHUNK_LENGTH_SECONDS = 30
MAX_AUDIO_FILE_SIZE_MB = 25
MAX_TRANSCRIPTION_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 3
DELAY_BETWEEN_REQUESTS_SECONDS = 1


# ---------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------

@dataclass
class AudioChunk:
    """
    Store information about one generated audio chunk.
    """

    path: Path
    start_seconds: float
    end_seconds: float


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
    Load the OpenAI API key and initialize the client.
    """
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found. "
            "Add it to the .env file."
        )

    if api_key == "your_openai_api_key_here":
        raise ValueError(
            "Replace the placeholder API key in the .env file."
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
    audio_files = sorted(
        file_path
        for file_path in AUDIO_DIR.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )

    if not audio_files:
        raise FileNotFoundError(
            "No supported audio file was found in the audio directory."
        )

    audio_file = audio_files[0]

    print(f"Audio file found: {audio_file.name}")

    return audio_file


def validate_audio_file(audio_file_path: Path) -> None:
    """
    Validate an audio file before processing.
    """
    if not audio_file_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_file_path}"
        )

    if not audio_file_path.is_file():
        raise ValueError(
            f"The audio path is not a file: {audio_file_path}"
        )

    if audio_file_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format: {audio_file_path.suffix}"
        )

    file_size_bytes = audio_file_path.stat().st_size

    if file_size_bytes == 0:
        raise ValueError(
            f"The audio file is empty: {audio_file_path.name}"
        )

    file_size_mb = file_size_bytes / (1024 * 1024)

    if file_size_mb > MAX_AUDIO_FILE_SIZE_MB:
        raise ValueError(
            f"{audio_file_path.name} is {file_size_mb:.2f} MB. "
            f"The maximum permitted size is "
            f"{MAX_AUDIO_FILE_SIZE_MB} MB."
        )

    print(
        f"Audio validation passed: "
        f"{audio_file_path.name} ({file_size_mb:.2f} MB)"
    )


def analyze_audio_file(
    audio_file_path: Path,
) -> AudioSegment:
    """
    Load the audio and print its properties.
    """
    audio = AudioSegment.from_file(audio_file_path)

    duration_seconds = len(audio) / 1000
    duration_minutes = duration_seconds / 60
    file_size_mb = (
        audio_file_path.stat().st_size / (1024 * 1024)
    )

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
# BASIC, UNGUIDED, AND GUIDED TRANSCRIPTION
# ---------------------------------------------------------

def transcribe_audio_basic(
    client: OpenAI,
    audio_file_path: Path,
) -> str:
    """
    Create a basic transcription without timestamps.
    """
    print("Starting basic transcription...")

    with audio_file_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )

    transcription_text = str(transcription).strip()

    if not transcription_text:
        raise ValueError(
            "The basic transcription returned no text."
        )

    print("Basic transcription completed successfully.")

    return transcription_text


def transcribe_audio_unguided(
    client: OpenAI,
    audio_file_path: Path,
) -> str:
    """
    Transcribe audio without contextual prompting.
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
    Transcribe audio using contextual prompting.
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
# TEXT OUTPUTS
# ---------------------------------------------------------

def save_text_transcription(
    transcription_text: str,
    output_file_name: str,
) -> Path:
    """
    Save a transcription to a TXT file.
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
    Save guided and unguided results in one file.
    """
    output_path = (
        OUTPUTS_DIR / "transcription_comparison.txt"
    )

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

    output_path.write_text(
        comparison_content,
        encoding="utf-8",
    )

    print(f"Comparison saved to: {output_path}")

    return output_path


# ---------------------------------------------------------
# AUDIO CHUNKING
# ---------------------------------------------------------

def clear_existing_chunks() -> None:
    """
    Remove previously generated chunk files.
    """
    for chunk_file in CHUNKS_DIR.glob("chunk_*.mp3"):
        chunk_file.unlink()

    print("Existing generated chunks cleared.")


def split_audio_into_chunks(
    audio: AudioSegment,
    chunk_length_seconds: int,
) -> list[AudioChunk]:
    """
    Split audio into fixed-length chunks.
    """
    if chunk_length_seconds <= 0:
        raise ValueError(
            "Chunk length must be greater than zero."
        )

    clear_existing_chunks()

    chunk_length_ms = chunk_length_seconds * 1000
    chunks: list[AudioChunk] = []

    print(
        f"Splitting audio into "
        f"{chunk_length_seconds}-second chunks..."
    )

    for chunk_number, start_ms in enumerate(
        range(0, len(audio), chunk_length_ms),
        start=1,
    ):
        end_ms = min(
            start_ms + chunk_length_ms,
            len(audio),
        )

        audio_chunk = audio[start_ms:end_ms]

        chunk_path = (
            CHUNKS_DIR
            / f"chunk_{chunk_number:03d}.mp3"
        )

        audio_chunk.export(
            chunk_path,
            format="mp3",
        )

        start_seconds = start_ms / 1000
        end_seconds = end_ms / 1000

        chunks.append(
            AudioChunk(
                path=chunk_path,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        )

        print(
            f"Created {chunk_path.name}: "
            f"{start_seconds:.2f}s to {end_seconds:.2f}s"
        )

    print(
        f"Audio chunking completed successfully. "
        f"Total chunks: {len(chunks)}"
    )

    return chunks


# ---------------------------------------------------------
# CHUNK TRANSCRIPTION WITH TIMESTAMPS
# ---------------------------------------------------------

def convert_response_to_dict(
    transcription: Any,
) -> dict[str, Any]:
    """
    Convert the OpenAI response object into a dictionary.
    """
    if hasattr(transcription, "model_dump"):
        return transcription.model_dump()

    if isinstance(transcription, dict):
        return transcription

    raise TypeError(
        "The transcription response could not be converted "
        "into a dictionary."
    )


def transcribe_chunk_with_timestamps(
    client: OpenAI,
    chunk: AudioChunk,
    prompt: str | None = None,
) -> list[dict[str, Any]]:
    """
    Transcribe one chunk and adjust its timestamps.
    """
    validate_audio_file(chunk.path)

    request_parameters: dict[str, Any] = {
        "model": "whisper-1",
        "response_format": "verbose_json",
        "language": "en",
        "timestamp_granularities": ["segment"],
    }

    if prompt:
        request_parameters["prompt"] = prompt

    last_error: Exception | None = None

    for attempt_number in range(
        1,
        MAX_TRANSCRIPTION_ATTEMPTS + 1,
    ):
        try:
            print(
                f"Transcribing {chunk.path.name} "
                f"(attempt {attempt_number}/"
                f"{MAX_TRANSCRIPTION_ATTEMPTS})..."
            )

            with chunk.path.open("rb") as audio_file:
                transcription = (
                    client.audio.transcriptions.create(
                        file=audio_file,
                        **request_parameters,
                    )
                )

            transcription_data = convert_response_to_dict(
                transcription
            )

            raw_segments = transcription_data.get(
                "segments",
                [],
            )

            if not raw_segments:
                raise ValueError(
                    f"No timestamp segments were returned for "
                    f"{chunk.path.name}."
                )

            adjusted_segments: list[dict[str, Any]] = []

            for segment in raw_segments:
                local_start = float(
                    segment.get("start", 0.0)
                )
                local_end = float(
                    segment.get("end", 0.0)
                )
                text = str(
                    segment.get("text", "")
                ).strip()

                if not text:
                    continue

                adjusted_segments.append(
                    {
                        "chunk": chunk.path.name,
                        "start": round(
                            local_start
                            + chunk.start_seconds,
                            3,
                        ),
                        "end": round(
                            local_end
                            + chunk.start_seconds,
                            3,
                        ),
                        "text": text,
                    }
                )

            if not adjusted_segments:
                raise ValueError(
                    f"No usable text segments were returned "
                    f"for {chunk.path.name}."
                )

            print(
                f"{chunk.path.name} completed. "
                f"Segments found: "
                f"{len(adjusted_segments)}"
            )

            return adjusted_segments

        except Exception as error:
            last_error = error

            print(
                f"Attempt {attempt_number} failed for "
                f"{chunk.path.name}: "
                f"{type(error).__name__}: {error}"
            )

            if attempt_number < MAX_TRANSCRIPTION_ATTEMPTS:
                print(
                    f"Waiting {RETRY_DELAY_SECONDS} seconds "
                    f"before retrying..."
                )

                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"Transcription failed for {chunk.path.name} "
        f"after {MAX_TRANSCRIPTION_ATTEMPTS} attempts."
    ) from last_error


def transcribe_all_chunks(
    client: OpenAI,
    chunks: list[AudioChunk],
    prompt: str | None = None,
) -> list[dict[str, Any]]:
    """
    Transcribe every chunk and combine all segments.
    """
    all_segments: list[dict[str, Any]] = []

    print("Starting chunked transcription with timestamps...")

    for chunk_index, chunk in enumerate(chunks):
        chunk_segments = transcribe_chunk_with_timestamps(
            client=client,
            chunk=chunk,
            prompt=prompt,
        )

        all_segments.extend(chunk_segments)

        if chunk_index < len(chunks) - 1:
            print(
                f"Waiting "
                f"{DELAY_BETWEEN_REQUESTS_SECONDS} second "
                f"before the next chunk..."
            )

            time.sleep(
                DELAY_BETWEEN_REQUESTS_SECONDS
            )

    if not all_segments:
        raise ValueError(
            "The chunked transcription returned no segments."
        )

    print(
        "Chunked transcription completed successfully. "
        f"Total segments: {len(all_segments)}"
    )

    return all_segments


# ---------------------------------------------------------
# TIMESTAMP FORMATTING
# ---------------------------------------------------------

def format_txt_timestamp(seconds: float) -> str:
    """
    Convert seconds into HH:MM:SS.
    """
    total_seconds = int(seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"
    )


def format_srt_timestamp(seconds: float) -> str:
    """
    Convert seconds into HH:MM:SS,mmm.
    """
    total_milliseconds = round(seconds * 1000)

    hours = total_milliseconds // 3_600_000
    remaining = total_milliseconds % 3_600_000

    minutes = remaining // 60_000
    remaining %= 60_000

    whole_seconds = remaining // 1000
    milliseconds = remaining % 1000

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{whole_seconds:02d},"
        f"{milliseconds:03d}"
    )


# ---------------------------------------------------------
# TIMESTAMP EXPORTS
# ---------------------------------------------------------

def export_segments_to_txt(
    segments: list[dict[str, Any]],
) -> Path:
    """
    Export timestamped segments to TXT.
    """
    output_path = (
        OUTPUTS_DIR / "chunked_transcription.txt"
    )

    lines: list[str] = []

    for segment in segments:
        start_time = format_txt_timestamp(
            float(segment["start"])
        )
        end_time = format_txt_timestamp(
            float(segment["end"])
        )

        lines.append(
            f"[{start_time} - {end_time}]"
        )
        lines.append(str(segment["text"]))
        lines.append("")

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"TXT transcription saved to: {output_path}")

    return output_path


def export_segments_to_json(
    segments: list[dict[str, Any]],
    source_audio: Path,
) -> Path:
    """
    Export timestamped segments to JSON.
    """
    output_path = (
        OUTPUTS_DIR / "chunked_transcription.json"
    )

    output_data = {
        "source_audio": source_audio.name,
        "model": "whisper-1",
        "language": "en",
        "chunk_length_seconds": CHUNK_LENGTH_SECONDS,
        "segment_count": len(segments),
        "segments": segments,
    }

    output_path.write_text(
        json.dumps(
            output_data,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"JSON transcription saved to: {output_path}")

    return output_path


def export_segments_to_srt(
    segments: list[dict[str, Any]],
) -> Path:
    """
    Export timestamped segments to SRT.
    """
    output_path = (
        OUTPUTS_DIR / "chunked_transcription.srt"
    )

    srt_blocks: list[str] = []

    for subtitle_number, segment in enumerate(
        segments,
        start=1,
    ):
        start_time = format_srt_timestamp(
            float(segment["start"])
        )
        end_time = format_srt_timestamp(
            float(segment["end"])
        )

        subtitle_block = (
            f"{subtitle_number}\n"
            f"{start_time} --> {end_time}\n"
            f"{segment['text']}"
        )

        srt_blocks.append(subtitle_block)

    output_path.write_text(
        "\n\n".join(srt_blocks) + "\n",
        encoding="utf-8",
    )

    print(f"SRT transcription saved to: {output_path}")

    return output_path


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

def main() -> None:
    """
    Run the complete Whisper transcription workflow.
    """
    print("=" * 60)
    print("WHISPER STT IMPLEMENTATION LAB")
    print("=" * 60)

    create_project_directories()

    try:
        client = load_openai_client()

        audio_file = find_audio_file()
        validate_audio_file(audio_file)
        audio = analyze_audio_file(audio_file)

        print("-" * 60)
        print("BASIC TRANSCRIPTION")
        print("-" * 60)

        basic_text = transcribe_audio_basic(
            client=client,
            audio_file_path=audio_file,
        )

        print(basic_text)

        save_text_transcription(
            transcription_text=basic_text,
            output_file_name="basic_transcription.txt",
        )

        time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)

        print("-" * 60)
        print("UNGUIDED TRANSCRIPTION")
        print("-" * 60)

        unguided_text = transcribe_audio_unguided(
            client=client,
            audio_file_path=audio_file,
        )

        print(unguided_text)

        save_text_transcription(
            transcription_text=unguided_text,
            output_file_name="unguided_transcription.txt",
        )

        time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)

        print("-" * 60)
        print("GUIDED TRANSCRIPTION")
        print("-" * 60)

        guided_text = transcribe_audio_guided(
            client=client,
            audio_file_path=audio_file,
            prompt=GUIDED_TRANSCRIPTION_PROMPT,
        )

        print(guided_text)

        save_text_transcription(
            transcription_text=guided_text,
            output_file_name="guided_transcription.txt",
        )

        save_comparison_file(
            unguided_text=unguided_text,
            guided_text=guided_text,
        )

        print("-" * 60)
        print("AUDIO CHUNKING")
        print("-" * 60)

        chunks = split_audio_into_chunks(
            audio=audio,
            chunk_length_seconds=CHUNK_LENGTH_SECONDS,
        )

        print("-" * 60)
        print("CHUNKED TRANSCRIPTION WITH TIMESTAMPS")
        print("-" * 60)

        timestamped_segments = transcribe_all_chunks(
            client=client,
            chunks=chunks,
            prompt=GUIDED_TRANSCRIPTION_PROMPT,
        )

        print("-" * 60)
        print("TIMESTAMPED SEGMENTS")
        print("-" * 60)

        for segment in timestamped_segments:
            start_time = format_txt_timestamp(
                float(segment["start"])
            )
            end_time = format_txt_timestamp(
                float(segment["end"])
            )

            print(
                f"[{start_time} - {end_time}] "
                f"{segment['text']}"
            )

        print("-" * 60)
        print("EXPORTING RESULTS")
        print("-" * 60)

        export_segments_to_txt(
            segments=timestamped_segments,
        )

        export_segments_to_json(
            segments=timestamped_segments,
            source_audio=audio_file,
        )

        export_segments_to_srt(
            segments=timestamped_segments,
        )

    except FileNotFoundError as error:
        print(f"File error: {error}")

    except ValueError as error:
        print(f"Configuration error: {error}")

    except TypeError as error:
        print(f"Data error: {error}")

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
# Whisper STT Implementation Lab

This project implements an audio transcription workflow using the OpenAI Whisper API.

The system can:

- Load and validate audio files
- Analyze audio properties
- Transcribe audio without prompts
- Transcribe audio with contextual prompts
- Compare guided and unguided transcription results
- Split long audio files into chunks
- Transcribe chunks with timestamps
- Adjust timestamps using chunk offsets
- Export results to TXT, JSON, and SRT
- Retry failed API requests
- Validate file size and supported formats

## Project Structure

```text
whisper_stt_implementation_lab/
│
├── audio/
│   └── CA138clip.mp3
│
├── chunks/
│   └── Generated audio chunks
│
├── outputs/
│   ├── basic_transcription.txt
│   ├── guided_transcription.txt
│   ├── unguided_transcription.txt
│   ├── transcription_comparison.txt
│   ├── chunked_transcription.txt
│   ├── chunked_transcription.json
│   └── chunked_transcription.srt
│
├── report/
│   └── report.md
│
├── screenshots/
│
├── .env
├── .gitignore
├── requirements.txt
├── README.md
└── whisper_transcription.py

Requirements
Python 3.12
OpenAI Python SDK
pydub
python-dotenv
FFmpeg
Installation

Create and activate the Conda environment:

conda create --name whisper-stt-env python=3.12 -y
conda activate whisper-stt-env

Install the required Python packages:

python -m pip install openai pydub python-dotenv

Verify FFmpeg:

ffmpeg -version
Environment Variables

Create a .env file in the root directory:

OPENAI_API_KEY=your_openai_api_key_here

Do not commit the .env file to GitHub.

Run the Project

On systems where invalid SSL environment variables are configured, clear them first:

unset SSL_CERT_FILE
unset REQUESTS_CA_BUNDLE

Run the script:

python whisper_transcription.py
Guided and Unguided Transcription

The unguided approach sends the audio without contextual information.

The guided approach includes a prompt describing the expected subject and terminology.

In the test audio, the unguided transcription produced:

get back to Maine

The guided transcription produced:

get back to Earth

The guided result better matched the context of the Moon mission.

However, prompts can also introduce incorrect interpretations, so human review is still necessary.

Audio Chunking

The sample audio is 86.20 seconds long.

For demonstration purposes, the script divides it into 30-second chunks:

chunk_001.mp3: 0.00s to 30.00s
chunk_002.mp3: 30.00s to 60.00s
chunk_003.mp3: 60.00s to 86.20s

For longer recordings, the chunk duration can be increased by changing:

CHUNK_LENGTH_SECONDS = 30
Timestamp Handling

Each chunk returns local timestamps starting near zero.

The script adds the chunk start offset to every segment:

global timestamp = local timestamp + chunk start offset

This ensures that timestamps continue across the full recording.

Output Formats
TXT

Human-readable transcription with timestamps:

[00:00:15 - 00:00:19]
I bet those men are going to get quite a reception when they get back to Earth.
JSON

Structured transcription data containing:

Source audio
Model
Language
Chunk length
Segment count
Chunk name
Start time
End time
Transcribed text
SRT

Subtitle-compatible output:

1
00:00:00,000 --> 00:00:12,000
It was rather interesting just to watch them gathering their materials...
Error Handling

The script includes:

Missing file validation
Empty file validation
Unsupported format validation
File size validation
Maximum file size of 25 MB
Automatic retries
Delay between API requests
Clear error messages
Main Technologies
Python
OpenAI Whisper API
pydub
FFmpeg
Git
GitHub
Conda
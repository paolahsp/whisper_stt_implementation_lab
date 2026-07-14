# Whisper STT Implementation Lab

This project implements an end-to-end audio transcription workflow using the OpenAI Whisper API.

The system can:

- Load and validate audio files
- Analyze audio properties
- Create a basic transcription
- Transcribe audio without prompts
- Transcribe audio with contextual prompts
- Compare guided and unguided results
- Split long audio files into chunks
- Extract segment timestamps
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
```

The `.env` file is used locally and is not committed to GitHub.

## Requirements

- Python 3.12
- OpenAI Python SDK
- pydub
- python-dotenv
- FFmpeg
- Conda
- Git

## Installation

Create and activate the Conda environment:

```bash
conda create --name whisper-stt-env python=3.12 -y
conda activate whisper-stt-env
```

Install the required Python packages:

```bash
python -m pip install openai pydub python-dotenv
```

Verify FFmpeg:

```bash
ffmpeg -version
```

## Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Do not commit the `.env` file to GitHub.

## Run the Project

If invalid SSL environment variables are configured, clear them before running:

```bash
unset SSL_CERT_FILE
unset REQUESTS_CA_BUNDLE
```

Run the project:

```bash
python whisper_transcription.py
```

## Basic Transcription

The basic transcription sends the audio to the Whisper API without timestamps or contextual guidance.

The result is saved to:

```text
outputs/basic_transcription.txt
```

## Guided and Unguided Transcription

The unguided approach sends the audio without contextual information.

The guided approach includes a prompt describing the expected subject and vocabulary.

In the sample audio, the unguided transcription produced:

```text
get back to Maine
```

The guided transcription produced:

```text
get back to Earth
```

The guided result better matched the Moon mission context.

However, prompting can also introduce incorrect interpretations. For example, one guided result changed:

```text
sending up ships
```

to:

```text
standing up
```

This shows that human review is still necessary.

## Audio Chunking

The sample audio is 86.20 seconds long.

For demonstration purposes, the script divides it into 30-second chunks:

```text
chunk_001.mp3: 0.00s to 30.00s
chunk_002.mp3: 30.00s to 60.00s
chunk_003.mp3: 60.00s to 86.20s
```

For longer recordings, the chunk duration can be changed in:

```python
CHUNK_LENGTH_SECONDS = 30
```

## Timestamp Handling

Each chunk returns local timestamps beginning near zero.

The script converts local timestamps into global timestamps using:

```text
global timestamp = local timestamp + chunk start offset
```

This creates one continuous timeline for the complete recording.

## Output Formats

### TXT

Human-readable transcription with timestamps:

```text
[00:00:15 - 00:00:19]
I bet those men are going to get quite a reception when they get back to Earth.
```

### JSON

Structured output containing:

- Source audio
- Model
- Language
- Chunk length
- Segment count
- Chunk name
- Start time
- End time
- Transcribed text

### SRT

Subtitle-compatible output:

```text
1
00:00:00,000 --> 00:00:12,000
It was rather interesting just to watch them gathering their materials...
```

## Error Handling

The script includes:

- Missing file validation
- Empty file validation
- Unsupported format validation
- File size validation
- Maximum file size validation
- Automatic retry handling
- Delay between API requests
- Clear error messages

## Security

The OpenAI API key is stored in `.env`.

The `.env` file is excluded through `.gitignore` and is not included in the repository.

## Main Technologies

- Python
- OpenAI Whisper API
- pydub
- FFmpeg
- Conda
- Git
- GitHub
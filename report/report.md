# Whisper STT Implementation Lab Report

## Overview

This project implemented a speech-to-text workflow using the OpenAI Whisper API. The system transcribed an English audio recording, compared guided and unguided transcription approaches, divided the recording into chunks, extracted timestamps, and exported the results in TXT, JSON, and SRT formats.

## Guided vs. Unguided Transcription

The unguided transcription processed the audio without contextual information. It successfully captured most of the conversation but produced some spacing and contextual errors. For example, it transcribed one phrase as “get back to Maine.”

The guided transcription included a prompt explaining that the recording discussed the Apollo Moon mission, astronauts, lunar exploration, rendezvous, orbiting, and soil samples. With this context, the model changed “Maine” to “Earth,” which was more accurate for the subject of the recording.

The guided version also improved spacing in phrases such as:

```text
I think that's
```

and:

```text
so I understand
```

However, the guided transcription was not perfect. It changed “sending up ships” to “standing up,” which may not accurately represent the original audio.

This demonstrated that prompts can improve contextual accuracy, but they can also influence the model incorrectly. Human review remains important.

## Benefits of Audio Chunking

Chunking makes it possible to process long recordings that may exceed API file-size limits.

It also reduces the impact of a failed request because only one part of the recording needs to be retried.

For this project, the 86.20-second audio file was divided into three chunks:

```text
chunk_001.mp3: 0.00s to 30.00s
chunk_002.mp3: 30.00s to 60.00s
chunk_003.mp3: 60.00s to 86.20s
```

Each chunk was processed separately, and its local timestamps were adjusted using the chunk start offset.

The main disadvantage was that sentences could be divided at chunk boundaries. For example, one segment ended with:

```text
Do you imagine them sending a...
```

and the following chunk continued with a separate phrase.

This shows that overlapping chunks or context carried between chunks could improve continuity.

## Timestamp Handling

Whisper returned local timestamps for each chunk.

The script converted them into global timestamps using:

```text
global start time = local start time + chunk offset
global end time = local end time + chunk offset
```

This prevented each chunk from restarting at zero and created one continuous timeline for the full recording.

## Challenges

The main technical challenges were:

- Configuring FFmpeg
- Resolving invalid SSL certificate environment variables
- Ensuring that VS Code used the correct Conda interpreter
- Preserving timestamps across chunk boundaries
- Maintaining context between separate chunks
- Reviewing prompt-induced transcription errors

## Recommendations

For production use, I would recommend:

- Using high-quality audio with minimal background noise
- Including important names and technical terms in the prompt
- Using larger chunks when file size permits
- Adding a small overlap between chunks
- Carrying context from one chunk into the next
- Reviewing guided transcriptions for prompt-induced errors
- Storing structured JSON output for search and downstream analysis
- Using SRT output for subtitles and accessibility
- Logging failed requests and retry attempts
- Protecting API keys through environment variables

## Conclusion

The lab demonstrated that Whisper can provide accurate and searchable speech-to-text results.

Prompts can improve context-sensitive terms, while chunking enables long recordings to be processed safely.

Timestamp adjustment and multiple export formats make the transcription useful for meetings, subtitles, search, accessibility, and documentation.
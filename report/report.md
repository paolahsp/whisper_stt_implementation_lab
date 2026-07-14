# Whisper STT Implementation Lab Report

## Overview

This project implemented a speech-to-text workflow using the OpenAI Whisper API. The system transcribed an English audio recording, compared guided and unguided transcription approaches, divided the recording into chunks, extracted timestamps, and exported the results in TXT, JSON, and SRT formats.

## Guided vs. Unguided Transcription

The unguided transcription processed the audio without contextual information. It successfully captured most of the conversation but produced some spacing and contextual errors. For example, it transcribed the phrase as “get back to Maine.”

The guided transcription included a prompt explaining that the recording discussed the Apollo Moon mission, astronauts, lunar exploration, rendezvous, orbiting, and soil samples. With this context, the model changed “Maine” to “Earth,” which was more accurate for the subject of the recording. It also improved spacing in phrases such as “I think that's” and “so I understand.”

However, the guided transcription was not perfect. It changed “sending up ships” to “standing up,” which may not accurately represent the original audio. This demonstrated that prompts can improve contextual accuracy but may also influence the model incorrectly. Human review remains important.

## Benefits of Audio Chunking

Chunking makes it possible to process long recordings that may exceed API file-size limits. It also reduces the impact of a failed request because only one part of the recording needs to be retried.

For this project, the 86.20-second audio file was divided into three 30-second chunks. Each chunk was processed separately, and its local timestamps were adjusted using the chunk start offset.

The main disadvantage was that sentences could be split at chunk boundaries. For example, one segment ended with “Do you imagine them sending a...” and the following chunk continued with a different phrase. This shows that overlapping chunks or context carried between chunks could improve continuity.

## Timestamp Handling

Whisper returned local timestamps for each chunk. The script converted them into global timestamps using the following logic:

```text
global start time = local start time + chunk offset
global end time = local end time + chunk offset

This prevented every chunk from restarting at zero and created one continuous timeline for the full recording.

Challenges

The main technical challenges were configuring FFmpeg, resolving invalid SSL certificate environment variables, ensuring that VS Code used the correct Conda interpreter, and preserving timestamps across chunk boundaries.

Another challenge was maintaining transcription context when audio was divided into separate chunks. Prompting improved some terminology, but the results still required review.

Recommendations

For production use, I would recommend:

Using high-quality audio with minimal background noise
Including important names and technical terms in the prompt
Using larger chunks for long recordings when file size allows
Adding a small overlap between chunks
Carrying context from one chunk into the next
Reviewing guided transcriptions for prompt-induced errors
Storing structured JSON output for search and downstream analysis
Using SRT output for subtitles and accessibility
Conclusion

The lab demonstrated that Whisper can provide accurate and searchable speech-to-text results. Prompts can improve context-sensitive terms, while chunking enables long recordings to be processed safely. Timestamp adjustment and multiple export formats make the transcription useful for meetings, subtitles, search, and documentation.
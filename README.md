# Emotional TTS

This project provides two batch pipelines to build a per-personality voice library and a runtime entry point to synthesize arbitrary text on demand, in English or Spanish, with one of seven emotions per turn.

## Main Scripts

- `voice_design_clone.py`
  Batch pipeline. Generates reference voices from instructions (VoiceDesign) and then clones the final target sentences.

- `voice_clone.py`
  Batch pipeline. Clones directly from uploaded reference audios (without VoiceDesign).

- `voice_runtime.py`
  Runtime entry point for per-turn use cases (e.g. a robot's dialogue loop). Loads the Base clone model once, reuses the on-disk references built by the batch pipelines, and synthesizes any `(text, language, emotion)` request on demand.

- `voice_personality_config.py`
  Central configuration for emotions, reference texts, target sentences, and personality folder name.

## Emotions Used

Emotion keys must match across all configuration dictionaries:

- `anger`
- `disgust`
- `fear`
- `happiness`
- `neutral`
- `sadness`
- `surprise`

## Requirements

- Linux (recommended)
- Python 3.10+
- NVIDIA GPU + CUDA (recommended)
- Access to Hugging Face model downloads

Dependencies are listed in `requirements.txt`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you use `attn_implementation="flash_attention_2"`, install a FlashAttention version compatible with your environment.

## Pipeline 1: VoiceDesign + Clone

Script: `voice_design_clone.py`

### What It Does

1. Loads VoiceDesign model: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`.
2. Loads Base model for cloning: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` or `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
3. For each emotion:
4. Generates EN and ES references from personality instructions.
5. Builds a voice-clone prompt from each reference.
6. Generates final EN and ES cloned outputs.

### Output For This Pipeline

This script uses one folder per personality:

```text
output/
  <personality_folder>/
    personality.txt
    clones/
      clone_en_<emotion>.wav
      clone_es_<emotion>.wav
    voice_design_clone_ref/
      voice_design_ref_en_<emotion>.wav
      voice_design_ref_es_<emotion>.wav
```

## Pipeline 2: Clone From Uploaded Audios

Script: `voice_clone.py`

### What It Does

1. Reads reference audios from `voice_clone_ref/`.
2. Validates that all expected files exist (14 total).
3. Builds clone prompts by emotion and language.
4. Generates final EN and ES outputs per emotion.
5. Copies the reference audios used into the output folder.
6. Saves metadata in `personality.txt`.

### Important: Reference Transcriptions

In `voice_clone.py`, `VOICE_CLONE_REFS_TEXT_EN_BY_EMOTION` and `VOICE_CLONE_REFS_TEXT_ES_BY_EMOTION`
provide the transcriptions of your reference audios.

You must write these transcriptions in `voice_personality_config.py`:

- `VOICE_CLONE_REFS_TEXT_EN_BY_EMOTION`
- `VOICE_CLONE_REFS_TEXT_ES_BY_EMOTION`

These texts are used by `create_voice_clone_prompt(...)` and should match what is spoken in the uploaded reference audios as closely as possible.
If they do not match, cloning quality can degrade.

### Exact Input Audio Names

Upload these files under `voice_clone_ref/`:

- `voice_clone_ref_en_anger.wav`
- `voice_clone_ref_en_disgust.wav`
- `voice_clone_ref_en_fear.wav`
- `voice_clone_ref_en_happiness.wav`
- `voice_clone_ref_en_neutral.wav`
- `voice_clone_ref_en_sadness.wav`
- `voice_clone_ref_en_surprise.wav`
- `voice_clone_ref_es_anger.wav`
- `voice_clone_ref_es_disgust.wav`
- `voice_clone_ref_es_fear.wav`
- `voice_clone_ref_es_happiness.wav`
- `voice_clone_ref_es_neutral.wav`
- `voice_clone_ref_es_sadness.wav`
- `voice_clone_ref_es_surprise.wav`

If any file is missing, the script raises `FileNotFoundError` with the missing file list.

### Output For This Pipeline

```text
output/
  <personality_folder>/
    personality.txt
    clones/
      clone_en_<emotion>.wav
      clone_es_<emotion>.wav
    voice_clone_ref/
      voice_clone_ref_en_<emotion>.wav
      voice_clone_ref_es_<emotion>.wav
```

## Runtime: Per-Turn Synthesis

Script: `voice_runtime.py`

### What It Does

1. Loads the Base clone model once: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.
2. Reads on-disk references from `output/<personality>/voice_clone_ref/` (default personality: `personality`). Run one of the batch pipelines first to produce them.
3. On each request, builds (and caches) a clone prompt for the requested `(language, emotion)` and synthesizes the text into a wav file.

The first request for a given `(language, emotion)` pays for prompt construction; subsequent requests with the same combo reuse the cached prompt and only run the clone forward pass.

### Inputs

- `text` — what to say.
- `language` — `English` | `Spanish` (also accepts `en` / `es`).
- `emotion` — one of `anger`, `disgust`, `fear`, `happiness`, `neutral`, `sadness`, `surprise`.
- `output` — destination wav path. Optional: if omitted, the audio is saved as `output/<personality>/runtime/<timestamp>_<emotion>_<lang>.wav`.

### Modes

One-shot (loads model, synthesizes one sentence, exits):

```bash
python voice_runtime.py --text "¡Hola, qué alegría verte!" --language Spanish --emotion happiness --output turn1.wav
```

Hot server (model stays loaded; one JSON request per stdin line, one JSON reply per stdout line):

```bash
python voice_runtime.py --serve
{"text":"Hola","language":"Spanish","emotion":"happiness","output":"t1.wav"}
{"text":"Wait, did you hear that?","language":"English","emotion":"fear"}
```

In-process (recommended when the dialogue loop is Python):

```python
from voice_runtime import EmotionalSpeaker

speaker = EmotionalSpeaker(personality="personality")
speaker.preload("Spanish", "happiness")  # optional, removes first-turn latency
speaker.say(text="Hola", language="Spanish", emotion="happiness")  # auto path
speaker.say(text="Hola", language="Spanish", emotion="happiness", output="t1.wav")
```

Switch personalities with `--personality other_personality` (CLI) or `EmotionalSpeaker(personality="...")` (Python).

## Configuration

Edit `voice_personality_config.py` to change:

- `DEFAULT_PERSONALITY_FOLDER`
- `BASE_IDENTITY`
- `PERSONALITY_TRAITS_BY_EMOTION`
- `VOICE_DESIGN_REF_TEXT_EN_BY_EMOTION`
- `VOICE_DESIGN_REF_TEXT_ES_BY_EMOTION`
- `VOICE_CLONE_REFS_TEXT_EN_BY_EMOTION`
- `VOICE_CLONE_REFS_TEXT_ES_BY_EMOTION`
- `DEFAULT_SENTENCES_EN_BY_EMOTION`
- `DEFAULT_SENTENCES_ES_BY_EMOTION`
- `DEFAULT_EMOTION_ORDER`

`voice_clone.py` and `voice_design_clone.py` use this configuration by default.

For `voice_design_clone.py`, use `VOICE_DESIGN_REF_TEXT_EN_BY_EMOTION` and `VOICE_DESIGN_REF_TEXT_ES_BY_EMOTION` to define per-emotion ref transcripts.

For `voice_clone.py`, use `VOICE_CLONE_REFS_TEXT_EN_BY_EMOTION` and `VOICE_CLONE_REFS_TEXT_ES_BY_EMOTION` to define per-emotion refs transcripts of uploaded audios.

## Execution

From the `emotional-tts/` folder:

```bash
python voice_design_clone.py
python voice_clone.py
```

To override the output folder name used by `voice_design_clone.py` or `voice_clone.py`:

```bash
python3 voice_design_clone.py --output_dir personality
python3 voice_clone.py --output_dir personality
```

The script also accepts a legacy freeform style:

```bash
python3 voice_design_clone.py -- output_dir personality 1
python3 voice_clone.py -- output_dir personality 1
```

Both commands save into `output/personality/`.

## Current Structure

```text
emotional-tts/
  README.md
  requirements.txt
  voice_design_clone.py
  voice_clone.py
  voice_runtime.py
  voice_personality_config.py
  voice_clone_ref/
  output/
```

## Common Issues

1. CUDA out-of-memory error
  Reduce GPU load or free GPU processes.

2. FlashAttention error
  Install a compatible `flash-attn` version or set `attn_implementation = "eager"`.

3. Emotion mapping validation error
  Ensure emotion keys are identical across all dictionaries.

4. Audio read/write error
  Check `soundfile` installation and required system audio libraries.

## Acknowledgments

This project builds on top of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), used via the `qwen-tts` package and the `Qwen/Qwen3-TTS-*` models on Hugging Face.

## Licenses

The project code is yours. External models and dependencies are governed by their own licenses.

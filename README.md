# Emotion-Aware Voice Design + Voice Cloning (Qwen3-TTS)

This project provides an end-to-end pipeline to:

1. Generate emotion-conditioned reference voices with **Qwen3-TTS VoiceDesign**.
2. Build reusable clone prompts from those references.
3. Synthesize final English and Spanish utterances with **Qwen3-TTS Base** while preserving emotion.

The main script is:

- `ray_voice_design_clone.py`

## Features

- Class-based, readable pipeline (`VoiceDesignClonePipeline`).
- Explicit emotion mapping via dictionaries to guarantee alignment:
  - `emotion -> instruction`
  - `emotion -> English sentence`
  - `emotion -> Spanish sentence`
- Built-in map consistency validation (`_validate_emotion_maps`).
- Automatic creation of output folders (`ref/`, `output/`).
- Emotion-specific output files for easier evaluation.

## Project Structure

```text
Qwen3-TT/
├── ray_voice_design_clone.py
├── requirements.txt
├── README.md
├── ref/                # Generated emotion-specific reference audios
└── output/             # Final cloned audios (EN/ES per emotion)
```

## Requirements

- Linux (recommended for GPU workflows)
- Python 3.10+
- NVIDIA GPU + CUDA (recommended for performance)
- Access to Hugging Face model download endpoints

Python dependencies are listed in `requirements.txt`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you want to keep `attn_implementation="flash_attention_2"`, install FlashAttention in your environment.

## Usage

Run from this folder:

```bash
python ray_voice_design_clone.py
```

The script will:

1. Load `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`.
2. Load `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.
3. For each emotion in `emotion_order`:
   - Generate reference audio in English and Spanish.
   - Build voice clone prompts.
   - Generate final cloned sentence in English and Spanish.

## Output Files

### Reference files (`ref/`)

- `voice_design_reference_en_<emotion>.wav`
- `voice_design_reference_es_<emotion>.wav`

### Final files (`output/`)

- `clone_en_<emotion>.wav`
- `clone_es_<emotion>.wav`

Example emotions:

- `anger`
- `disgust`
- `fear`
- `happiness`
- `neutral`
- `sadness`
- `surprise`

## Configuration

Edit `build_default_data()` in `ray_voice_design_clone.py` to customize:

- `base_identity`
- `ref_instruct_by_emotion`
- `ref_text_en` / `ref_text_es`
- `sentences_en_by_emotion` / `sentences_es_by_emotion`
- `emotion_order`

Edit `ModelConfig` to customize runtime/model behavior:

- `voice_design_model_id`
- `clone_model_id`
- `device_map`
- `dtype`
- `attn_implementation`

## Notes on Model/Runtime

- Current defaults assume CUDA (`device_map="cuda:0"`).
- If FlashAttention is not installed, either install it or change:

```python
attn_implementation = "eager"
```

- First run may take longer due to model downloads and cache initialization.

## Troubleshooting

### 1) CUDA out of memory

- Reduce model size (if available).
- Close other GPU-heavy processes.
- Try lower memory settings or a different attention backend.

### 2) FlashAttention errors

- Install `flash-attn` compatible with your CUDA/PyTorch versions.
- Or set `attn_implementation` to `"eager"`.

### 3) Emotion mapping validation error

If you get errors like `keys must match emotion_order`, ensure all emotion dictionaries use exactly the same keys as `emotion_order`.

### 4) Audio writing issues

Ensure `soundfile` is installed correctly and your environment has required system audio libraries.

## Reproducibility Tips

- Keep `requirements.txt` pinned (already included).
- Commit exact script changes together with output naming conventions.
- Document GPU, CUDA, and driver versions in your GitHub release notes.

## License

This repository content is your project code. The referenced models and third-party packages are governed by their own licenses:

- Qwen3-TTS model/package licenses
- PyTorch, Transformers, and other dependency licenses

Review each upstream license before production or commercial use.

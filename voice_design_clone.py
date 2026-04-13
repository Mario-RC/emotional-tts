import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from voice_personality_config import build_default_generation_config


# Global model/runtime configuration used by both generation stages.
@dataclass(frozen=True)
class ModelConfig:
    voice_design_model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    clone_model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"  # "Qwen3-TTS-12Hz-0.6B-Base"
    device_map: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    attn_implementation: str = "flash_attention_2"
    voice_design_seed: int = 42


# Data contract for emotion-conditioned generation.
@dataclass(frozen=True)
class GenerationData:
    # Defines the canonical order used when iterating over emotions.
    emotion_order: List[str]
    # Maps each emotion to language-specific style instructions used to build references.
    ref_instruct_en_by_emotion: Dict[str, str]
    ref_instruct_es_by_emotion: Dict[str, str]
    # Per-emotion reference transcripts used in design and clone prompt creation.
    voice_design_ref_text_en_by_emotion: Dict[str, str]
    voice_design_ref_text_es_by_emotion: Dict[str, str]
    # Per-emotion target sentences for final cloned outputs.
    sentences_en_by_emotion: Dict[str, str]
    sentences_es_by_emotion: Dict[str, str]
    # Output folder label for this personality preset.
    personality_folder: str


class VoiceDesignClonePipeline:
    # Orchestrates end-to-end generation: reference creation, prompt building, and cloning.
    def __init__(self, config: ModelConfig, data: GenerationData) -> None:
        self.config = config
        self.data = data
        self.output_root_dir = Path("output")
        self.personality_dir = self.output_root_dir / self.data.personality_folder
        self.ref_dir = self.personality_dir / "voice_design_clone_ref"
        self.clone_dir = self.personality_dir / "clones"

    def run(self) -> None:
        # Prepare filesystem and fail fast if emotion maps are inconsistent.
        self._ensure_output_dirs()
        self._validate_emotion_maps()
        self._write_personality_file()

        # Load models once to avoid repeated initialization overhead.
        design_model = self._load_voice_design_model()
        clone_model = self._load_clone_model()

        # Process each emotion independently to keep prompt/style alignment explicit.
        for emotion in self.data.emotion_order:
            emotion_tag = self._file_tag(emotion)
            ref_instruct_en = self.data.ref_instruct_en_by_emotion[emotion]
            ref_instruct_es = self.data.ref_instruct_es_by_emotion[emotion]
            ref_text_en = self.data.voice_design_ref_text_en_by_emotion[emotion]
            ref_text_es = self.data.voice_design_ref_text_es_by_emotion[emotion]

            # Create emotion-specific reference audios in EN and ES with the same style instruction.
            ref_wav_en, sr = self._generate_ref_audio(
                model=design_model,
                text=ref_text_en,
                language="English",
                instruct=ref_instruct_en,
                out_path=self.ref_dir / f"voice_design_ref_en_{emotion_tag}.wav",
            )
            ref_wav_es, _ = self._generate_ref_audio(
                model=design_model,
                text=ref_text_es,
                language="Spanish",
                instruct=ref_instruct_es,
                out_path=self.ref_dir / f"voice_design_ref_es_{emotion_tag}.wav",
            )

            # Build clone prompts from the generated references.
            voice_clone_prompt_en = self._build_clone_prompt(
                clone_model=clone_model,
                ref_wav=ref_wav_en,
                sample_rate=sr,
                ref_text=ref_text_en,
            )
            voice_clone_prompt_es = self._build_clone_prompt(
                clone_model=clone_model,
                ref_wav=ref_wav_es,
                sample_rate=sr,
                ref_text=ref_text_es,
            )

            # Generate one final cloned sentence per language for the same emotion.
            self._generate_single_clone(
                clone_model=clone_model,
                text=self.data.sentences_en_by_emotion[emotion],
                language="English",
                voice_clone_prompt=voice_clone_prompt_en,
                out_file=self.clone_dir / f"clone_en_{emotion_tag}.wav",
            )
            self._generate_single_clone(
                clone_model=clone_model,
                text=self.data.sentences_es_by_emotion[emotion],
                language="Spanish",
                voice_clone_prompt=voice_clone_prompt_es,
                out_file=self.clone_dir / f"clone_es_{emotion_tag}.wav",
            )

    def _ensure_output_dirs(self) -> None:
        # Ensure personality folders exist before any write operation.
        self.output_root_dir.mkdir(parents=True, exist_ok=True)
        self.ref_dir.mkdir(parents=True, exist_ok=True)
        self.clone_dir.mkdir(parents=True, exist_ok=True)

    def _write_personality_file(self) -> None:
        # Persist the exact personality instructions used for this execution.
        personality_file = self.personality_dir / "personality.txt"
        lines = [
            "pipeline: voice_design_clone",
            f"personality_folder: {self.data.personality_folder}",
            f"output_folder: {self.personality_dir}",
            f"clone_folder: {self.clone_dir}",
            f"ref_folder: {self.ref_dir}",
            "instructions_by_emotion:",
        ]
        for emotion in self.data.emotion_order:
            ref_instruct_en = self.data.ref_instruct_en_by_emotion[emotion]
            ref_instruct_es = self.data.ref_instruct_es_by_emotion[emotion]
            lines.append(f"- {emotion} [English]: {ref_instruct_en}")
            lines.append(f"- {emotion} [Spanish]: {ref_instruct_es}")
        personality_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _load_voice_design_model(self) -> Qwen3TTSModel:
        # Model used to synthesize style reference audios from text + instruction.
        return Qwen3TTSModel.from_pretrained(
            self.config.voice_design_model_id,
            device_map=self.config.device_map,
            dtype=self.config.dtype,
            attn_implementation=self.config.attn_implementation,
        )

    def _load_clone_model(self) -> Qwen3TTSModel:
        # Model used to clone voice style from references into target sentences.
        return Qwen3TTSModel.from_pretrained(
            self.config.clone_model_id,
            device_map=self.config.device_map,
            dtype=self.config.dtype,
            attn_implementation=self.config.attn_implementation,
        )

    def _generate_ref_audio(
        self,
        model: Qwen3TTSModel,
        text: str,
        language: str,
        instruct: str,
        out_path: Path,
    ) -> Tuple[np.ndarray, int]:
        # Generate one reference waveform and persist it for traceability/reuse.
        self._set_voice_design_seed()
        wavs, sample_rate = model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
        )
        sf.write(str(out_path), wavs[0], sample_rate)
        return wavs[0], sample_rate

    def _set_voice_design_seed(self) -> None:
        # Freeze randomness right before voice design generation for reproducibility.
        seed_value = self.config.voice_design_seed
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)

    def _build_clone_prompt(
        self,
        clone_model: Qwen3TTSModel,
        ref_wav: np.ndarray,
        sample_rate: int,
        ref_text: str,
    ):
        # Build a reusable prompt object from reference audio and matching transcript.
        return clone_model.create_voice_clone_prompt(
            ref_audio=(ref_wav, sample_rate),
            ref_text=ref_text,
        )

    def _generate_single_clone(
        self,
        clone_model: Qwen3TTSModel,
        text: str,
        language: str,
        voice_clone_prompt,
        out_file: Path,
    ) -> None:
        # Run voice cloning for a single text and save the resulting waveform.
        wavs, sample_rate = clone_model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
        )
        sf.write(str(out_file), wavs[0], sample_rate)

    def _validate_emotion_maps(self) -> None:
        # All emotion-indexed maps must share exactly the same key set.
        base = set(self.data.emotion_order)
        if set(self.data.ref_instruct_en_by_emotion.keys()) != base:
            raise ValueError("ref_instruct_en_by_emotion keys must match emotion_order")
        if set(self.data.ref_instruct_es_by_emotion.keys()) != base:
            raise ValueError("ref_instruct_es_by_emotion keys must match emotion_order")
        if set(self.data.voice_design_ref_text_en_by_emotion.keys()) != base:
            raise ValueError("voice_design_ref_text_en_by_emotion keys must match emotion_order")
        if set(self.data.voice_design_ref_text_es_by_emotion.keys()) != base:
            raise ValueError("voice_design_ref_text_es_by_emotion keys must match emotion_order")
        if set(self.data.sentences_en_by_emotion.keys()) != base:
            raise ValueError("sentences_en_by_emotion keys must match emotion_order")
        if set(self.data.sentences_es_by_emotion.keys()) != base:
            raise ValueError("sentences_es_by_emotion keys must match emotion_order")

    @staticmethod
    def _file_tag(emotion: str) -> str:
        # Normalize emotion names for file-safe, stable output names.
        return emotion.strip().lower().replace(" ", "_")


def build_default_data() -> GenerationData:
    # Build generation defaults from a centralized config module.
    generation_config = build_default_generation_config()
    return GenerationData(
        emotion_order=generation_config["emotion_order"],
        ref_instruct_en_by_emotion=generation_config["ref_instruct_en_by_emotion"],
        ref_instruct_es_by_emotion=generation_config["ref_instruct_es_by_emotion"],
        voice_design_ref_text_en_by_emotion=generation_config[
            "voice_design_ref_text_en_by_emotion"
        ],
        voice_design_ref_text_es_by_emotion=generation_config[
            "voice_design_ref_text_es_by_emotion"
        ],
        sentences_en_by_emotion=generation_config["sentences_en_by_emotion"],
        sentences_es_by_emotion=generation_config["sentences_es_by_emotion"],
        personality_folder=generation_config["personality_folder"],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        nargs="+",
        help=(
            "Output folder name under output/. "
            "Example: --output_dir personality_1"
        ),
    )
    # Accepts legacy freeform usage like: python3 voice_design_clone.py -- output_dir personality 3
    parser.add_argument("legacy_output_dir", nargs="*")
    return parser.parse_args()


def _resolve_output_folder(args: argparse.Namespace) -> str | None:
    tokens: List[str] = []
    if args.output_dir:
        tokens = args.output_dir
    elif args.legacy_output_dir:
        legacy_tokens = args.legacy_output_dir
        if legacy_tokens[0] in {"output_dir", "output-dir", "--output_dir", "--output-dir"}:
            tokens = legacy_tokens[1:]
        else:
            tokens = legacy_tokens

    if not tokens:
        return None

    clean_parts = [part.strip() for part in tokens if part.strip()]
    if not clean_parts:
        return None
    return "_".join(clean_parts)


def main() -> None:
    # Entry point used when running this file as a script.
    args = _parse_args()
    data = build_default_data()
    output_folder_override = _resolve_output_folder(args)
    if output_folder_override:
        data = replace(data, personality_folder=output_folder_override)

    pipeline = VoiceDesignClonePipeline(config=ModelConfig(), data=data)
    pipeline.run()


if __name__ == "__main__":
    main()
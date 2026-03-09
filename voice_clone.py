import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from shutil import copy2
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from voice_personality_config import build_default_generation_config


@dataclass(frozen=True)
class ModelConfig:
    clone_model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"  # "Qwen3-TTS-12Hz-0.6B-Base"
    device_map: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    attn_implementation: str = "flash_attention_2"


@dataclass(frozen=True)
class GenerationData:
    emotion_order: List[str]
    voice_clone_refs_text_en_by_emotion: Dict[str, str]
    voice_clone_refs_text_es_by_emotion: Dict[str, str]
    sentences_en_by_emotion: Dict[str, str]
    sentences_es_by_emotion: Dict[str, str]
    personality_folder: str


class UploadedRefsClonePipeline:
    def __init__(self, config: ModelConfig, data: GenerationData) -> None:
        self.config = config
        self.data = data
        self.input_ref_dir = Path("voice_clone_ref")
        self.output_root_dir = Path("output")
        self.personality_dir = self.output_root_dir / self.data.personality_folder
        self.output_ref_dir = self.personality_dir / "voice_clone_ref"
        self.clone_dir = self.personality_dir / "clones"

    def run(self) -> None:
        self._ensure_output_dirs()
        self._validate_emotion_maps()
        self._validate_ref_files()
        self._copy_ref_files_to_run_folder()
        self._write_personality_file()

        clone_model = self._load_clone_model()

        for emotion in self.data.emotion_order:
            emotion_tag = self._file_tag(emotion)
            ref_text_en = self.data.voice_clone_refs_text_en_by_emotion[emotion]
            ref_text_es = self.data.voice_clone_refs_text_es_by_emotion[emotion]

            ref_wav_en, sr_en = self._read_ref_audio(
                self.input_ref_dir / f"voice_clone_ref_en_{emotion_tag}.wav"
            )
            ref_wav_es, sr_es = self._read_ref_audio(
                self.input_ref_dir / f"voice_clone_ref_es_{emotion_tag}.wav"
            )

            voice_clone_prompt_en = clone_model.create_voice_clone_prompt(
                ref_audio=(ref_wav_en, sr_en),
                ref_text=ref_text_en,
            )
            voice_clone_prompt_es = clone_model.create_voice_clone_prompt(
                ref_audio=(ref_wav_es, sr_es),
                ref_text=ref_text_es,
            )

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
        self.output_root_dir.mkdir(parents=True, exist_ok=True)
        self.output_ref_dir.mkdir(parents=True, exist_ok=True)
        self.personality_dir.mkdir(parents=True, exist_ok=True)
        self.clone_dir.mkdir(parents=True, exist_ok=True)

    def _validate_emotion_maps(self) -> None:
        base = set(self.data.emotion_order)
        if set(self.data.voice_clone_refs_text_en_by_emotion.keys()) != base:
            raise ValueError("voice_clone_refs_text_en_by_emotion keys must match emotion_order")
        if set(self.data.voice_clone_refs_text_es_by_emotion.keys()) != base:
            raise ValueError("voice_clone_refs_text_es_by_emotion keys must match emotion_order")
        if set(self.data.sentences_en_by_emotion.keys()) != base:
            raise ValueError("sentences_en_by_emotion keys must match emotion_order")
        if set(self.data.sentences_es_by_emotion.keys()) != base:
            raise ValueError("sentences_es_by_emotion keys must match emotion_order")

    def _validate_ref_files(self) -> None:
        missing_files: List[str] = []
        for emotion in self.data.emotion_order:
            emotion_tag = self._file_tag(emotion)
            en_path = self.input_ref_dir / f"voice_clone_ref_en_{emotion_tag}.wav"
            es_path = self.input_ref_dir / f"voice_clone_ref_es_{emotion_tag}.wav"
            if not en_path.exists():
                missing_files.append(str(en_path))
            if not es_path.exists():
                missing_files.append(str(es_path))

        if missing_files:
            missing_text = "\n".join(missing_files)
            raise FileNotFoundError(
                "Missing reference audios. Upload the files to 'voice_clone_ref/' with these names:\n"
                f"{missing_text}"
            )

    def _copy_ref_files_to_run_folder(self) -> None:
        for emotion in self.data.emotion_order:
            emotion_tag = self._file_tag(emotion)
            for language in ("en", "es"):
                source = self.input_ref_dir / f"voice_clone_ref_{language}_{emotion_tag}.wav"
                destination = self.output_ref_dir / source.name
                copy2(source, destination)

    def _write_personality_file(self) -> None:
        personality_file = self.personality_dir / "personality.txt"
        lines = [
            "pipeline: voice_clone",
            f"personality_folder: {self.data.personality_folder}",
            f"output_folder: {self.personality_dir}",
            f"clone_folder: {self.clone_dir}",
            f"ref_folder: {self.output_ref_dir}",
            f"input_ref_dir: {self.input_ref_dir}",
            "ref_en_files_by_emotion:",
        ]

        for emotion in self.data.emotion_order:
            emotion_tag = self._file_tag(emotion)
            en_name = f"voice_clone_ref_en_{emotion_tag}.wav"
            lines.append(f"- {emotion}: {en_name}")

        lines.append("ref_es_files_by_emotion:")
        for emotion in self.data.emotion_order:
            emotion_tag = self._file_tag(emotion)
            es_name = f"voice_clone_ref_es_{emotion_tag}.wav"
            lines.append(f"- {emotion}: {es_name}")

        lines.append("ref_texts_by_emotion:")
        for emotion in self.data.emotion_order:
            ref_text_en = self.data.voice_clone_refs_text_en_by_emotion[emotion]
            ref_text_es = self.data.voice_clone_refs_text_es_by_emotion[emotion]
            lines.append(
                f"- {emotion}: ref_text_en={ref_text_en}, ref_text_es={ref_text_es}"
            )

        personality_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _load_clone_model(self) -> Qwen3TTSModel:
        return Qwen3TTSModel.from_pretrained(
            self.config.clone_model_id,
            device_map=self.config.device_map,
            dtype=self.config.dtype,
            attn_implementation=self.config.attn_implementation,
        )

    def _read_ref_audio(self, audio_path: Path) -> tuple[np.ndarray, int]:
        wav, sample_rate = sf.read(str(audio_path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav, sample_rate

    def _generate_single_clone(
        self,
        clone_model: Qwen3TTSModel,
        text: str,
        language: str,
        voice_clone_prompt,
        out_file: Path,
    ) -> None:
        wavs, sample_rate = clone_model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
        )
        sf.write(str(out_file), wavs[0], sample_rate)

    @staticmethod
    def _file_tag(emotion: str) -> str:
        return emotion.strip().lower().replace(" ", "_")


def build_default_data() -> GenerationData:
    generation_config = build_default_generation_config()
    return GenerationData(
        emotion_order=generation_config["emotion_order"],
        voice_clone_refs_text_en_by_emotion=generation_config[
            "voice_clone_refs_text_en_by_emotion"
        ],
        voice_clone_refs_text_es_by_emotion=generation_config[
            "voice_clone_refs_text_es_by_emotion"
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
            "Example: --output_dir personality_3"
        ),
    )
    # Accepts legacy freeform usage like: python3 voice_clone.py -- output_dir personality 3
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
    args = _parse_args()
    data = build_default_data()
    output_folder_override = _resolve_output_folder(args)
    if output_folder_override:
        data = replace(data, personality_folder=output_folder_override)

    pipeline = UploadedRefsClonePipeline(config=ModelConfig(), data=data)
    pipeline.run()


if __name__ == "__main__":
    main()

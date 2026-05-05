"""Hot emotional TTS for per-turn robot speech.

Reuses the reference audios already generated under a personality folder
(default: output/personality/voice_clone_ref/) so only the Base clone model
is loaded — no VoiceDesign pass at runtime.

Two modes:

- One-shot:
    python voice_runtime.py --text "Hola" --language Spanish --emotion happiness --output out.wav

- Hot server (model stays loaded, read JSON requests from stdin):
    python voice_runtime.py --serve
    > {"text":"Hola","language":"Spanish","emotion":"happiness","output":"out.wav"}
    > {"text":"Goodbye","language":"English","emotion":"sadness","output":"bye.wav"}

Or import in process:
    from voice_runtime import EmotionalSpeaker
    speaker = EmotionalSpeaker()
    speaker.say(text="Hola", language="Spanish", emotion="happiness", output="out.wav")

If `output` is omitted, a unique wav is written under
`output/<personality>/runtime/`.

The first call for a given (language, emotion) builds and caches the clone
prompt from the on-disk reference; subsequent calls reuse it, so per-turn
latency drops to a single voice-clone forward pass.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from voice_personality_config import (
    PERSONALITY_TRAITS_BY_EMOTION,
    VOICE_CLONE_REFS_TEXT_EN_BY_EMOTION,
    VOICE_CLONE_REFS_TEXT_ES_BY_EMOTION,
)


CLONE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEVICE_MAP = "cuda:0"
DTYPE = torch.bfloat16
ATTN_IMPL = "flash_attention_2"

DEFAULT_PERSONALITY = "personality"
DEFAULT_OUTPUT_ROOT = Path("output")
RUNTIME_SUBDIR = "runtime"

# Map free-form language inputs to the canonical names Qwen3-TTS expects.
LANG_NAMES = {
    "en": "English", "english": "English",
    "es": "Spanish", "spanish": "Spanish", "español": "Spanish", "espanol": "Spanish",
}
# Short codes used in the on-disk reference filenames.
LANG_TO_CODE = {"English": "en", "Spanish": "es"}
REF_TEXT_BY_LANG = {
    "English": VOICE_CLONE_REFS_TEXT_EN_BY_EMOTION,
    "Spanish": VOICE_CLONE_REFS_TEXT_ES_BY_EMOTION,
}


class EmotionalSpeaker:
    """Loads Qwen3-TTS Base once and caches per-(language, emotion) clone prompts."""

    def __init__(
        self,
        personality: str = DEFAULT_PERSONALITY,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
    ) -> None:
        self._personality = personality
        self._output_root = Path(output_root)
        self._ref_dir = self._output_root / personality / "voice_clone_ref"
        self._runtime_dir = self._output_root / personality / RUNTIME_SUBDIR
        if not self._ref_dir.is_dir():
            raise FileNotFoundError(f"Reference folder not found: {self._ref_dir}")

        self._clone = Qwen3TTSModel.from_pretrained(
            CLONE_MODEL_ID,
            device_map=DEVICE_MAP,
            dtype=DTYPE,
            attn_implementation=ATTN_IMPL,
        )
        self._prompt_cache: Dict[Tuple[str, str], object] = {}

    def say(
        self,
        text: str,
        language: str,
        emotion: str,
        output: Optional[str] = None,
    ) -> str:
        lang = self._normalize_lang(language)
        if emotion not in PERSONALITY_TRAITS_BY_EMOTION:
            allowed = sorted(PERSONALITY_TRAITS_BY_EMOTION)
            raise ValueError(f"Unknown emotion '{emotion}'. Allowed: {allowed}")

        prompt = self._get_clone_prompt(lang, emotion)
        wavs, sample_rate = self._clone.generate_voice_clone(
            text=text,
            language=lang,
            voice_clone_prompt=prompt,
        )
        out_path = self._resolve_output(output, lang, emotion)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), wavs[0], sample_rate)
        return str(out_path)

    def preload(self, language: str, emotion: str) -> None:
        # Eagerly build and cache a clone prompt to remove first-turn latency.
        self._get_clone_prompt(self._normalize_lang(language), emotion)

    def _resolve_output(self, output: Optional[str], language: str, emotion: str) -> Path:
        if output:
            return Path(output)
        # Nanosecond timestamp keeps successive turns from clobbering each other.
        timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
        code = LANG_TO_CODE[language]
        return self._runtime_dir / f"{timestamp}_{emotion}_{code}.wav"

    def _get_clone_prompt(self, language: str, emotion: str):
        key = (language, emotion)
        cached = self._prompt_cache.get(key)
        if cached is not None:
            return cached

        ref_wav, sample_rate = self._read_ref_audio(language, emotion)
        ref_text = REF_TEXT_BY_LANG[language][emotion]
        prompt = self._clone.create_voice_clone_prompt(
            ref_audio=(ref_wav, sample_rate), ref_text=ref_text,
        )
        self._prompt_cache[key] = prompt
        return prompt

    def _read_ref_audio(self, language: str, emotion: str) -> Tuple[np.ndarray, int]:
        code = LANG_TO_CODE[language]
        ref_path = self._ref_dir / f"voice_clone_ref_{code}_{emotion}.wav"
        if not ref_path.is_file():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")
        wav, sample_rate = sf.read(str(ref_path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav, sample_rate

    @staticmethod
    def _normalize_lang(language: str) -> str:
        key = language.strip().lower()
        canonical = LANG_NAMES.get(key)
        if canonical is None:
            raise ValueError(f"Unsupported language '{language}'. Use 'English' or 'Spanish'.")
        return canonical


def _serve(speaker: EmotionalSpeaker) -> None:
    # Line-delimited JSON protocol: one request per stdin line, one reply per stdout line.
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            path = speaker.say(
                text=req["text"],
                language=req["language"],
                emotion=req["emotion"],
                output=req.get("output"),
            )
            print(json.dumps({"ok": True, "output": path}), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hot emotional TTS for per-turn robot speech.")
    parser.add_argument("--text", help="Text to synthesize (one-shot mode).")
    parser.add_argument("--language", help="English | Spanish | en | es (one-shot mode).")
    parser.add_argument(
        "--emotion",
        help=f"One of: {sorted(PERSONALITY_TRAITS_BY_EMOTION)} (one-shot mode).",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output wav path. If omitted, writes to "
            f"output/<personality>/{RUNTIME_SUBDIR}/<timestamp>_<emotion>_<lang>.wav"
        ),
    )
    parser.add_argument(
        "--personality",
        default=DEFAULT_PERSONALITY,
        help=f"Personality folder under output/ (default: {DEFAULT_PERSONALITY}).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Stay loaded and read JSON requests from stdin (one per line).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    speaker = EmotionalSpeaker(personality=args.personality)

    if args.serve:
        _serve(speaker)
        return

    missing = [name for name in ("text", "language", "emotion") if not getattr(args, name)]
    if missing:
        sys.exit(f"Missing required args: {missing}. Use --serve for hot mode.")
    print(speaker.say(args.text, args.language, args.emotion, args.output))


if __name__ == "__main__":
    main()

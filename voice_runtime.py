"""Hot emotional TTS for per-turn robot speech.

Reuses the reference audios already generated under a personality folder
(default: artifacts/personality/ref/voice_design_clone_ref/) so only the Base clone model
is loaded — no VoiceDesign pass at runtime.

Two modes:

- One-shot:
    python voice_runtime.py --text "Hola" --language Spanish --emotion happiness --output voice_runtime_es_happiness.wav

- Hot server (model stays loaded, read JSON requests from stdin):
    python voice_runtime.py --serve --preload-all --warmup
    > {"text":"Hola","language":"Spanish","emotion":"happiness","output":"out.wav"}
    > {"texts":["Hola.","Qué alegría verte.","Vamos allá."],"language":"Spanish","emotion":"happiness"}
    > {"text":"Goodbye","language":"English","emotion":"sadness","output":"bye.wav"}

Or import in process:
    from voice_runtime import EmotionalSpeaker
    speaker = EmotionalSpeaker()
    speaker.say(text="Hola", language="Spanish", emotion="happiness", output="out.wav")

If `output` is omitted, a normalized wav is written under
`artifacts/<personality>/generated_speech/voice_runtime/`.
If `output` is a relative path, it is resolved inside that same folder.

The first call for a given (language, emotion) builds and caches the clone
prompt from the on-disk reference; subsequent calls reuse it, so per-turn
latency drops to a single voice-clone forward pass.

For low-latency dialogue loops, run `--serve --preload-all --warmup`.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from utils import SilenceTrimConfig, trim_silence
from voice_personality_config import (
    PERSONALITY_TRAITS_BY_EMOTION,
    VOICE_DESIGN_REF_TEXT_EN_BY_EMOTION,
    VOICE_DESIGN_REF_TEXT_ES_BY_EMOTION,
)


CLONE_MODEL_IDS = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}
DEFAULT_MODEL_SIZE = "0.6B"
DEFAULT_CLONE_MODEL_ID = CLONE_MODEL_IDS[DEFAULT_MODEL_SIZE]
DEVICE_MAP = "cuda:0"
DTYPE = torch.bfloat16
ATTN_IMPL = "flash_attention_2"

DEFAULT_PERSONALITY = "personality"
DEFAULT_OUTPUT_ROOT = Path("artifacts")
REF_SUBDIR = Path("ref") / "voice_design_clone_ref"
RUNTIME_SUBDIR = Path("generated_speech") / "voice_runtime"

# Map free-form language inputs to the canonical names Qwen3-TTS expects.
LANG_NAMES = {
    "en": "English", "english": "English",
    "es": "Spanish", "spanish": "Spanish", "español": "Spanish", "espanol": "Spanish",
}
# Short codes used in the on-disk reference filenames.
LANG_TO_CODE = {"English": "en", "Spanish": "es"}
DEFAULT_REF_TEXT_BY_LANG = {
    "English": VOICE_DESIGN_REF_TEXT_EN_BY_EMOTION,
    "Spanish": VOICE_DESIGN_REF_TEXT_ES_BY_EMOTION,
}
REF_FILE_PREFIXES = ("voice_design_ref", "voice_clone_ref")
DEFAULT_LATENCY_PRESET = "fast"
MAX_BATCH_TEXTS = 3
COMBINED_OUTPUT_PAUSE_MS = 250
MIN_MAX_NEW_TOKENS = 96
MIN_MAX_NEW_TOKENS_PER_CHAR = 4
MAX_DYNAMIC_MAX_NEW_TOKENS = 2048
CLONED_AUDIO_TRIM_CONFIG = SilenceTrimConfig(
    enabled=True,
    threshold=0.003,
    padding_ms=80,
)
LATENCY_PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "do_sample": True,
        "top_k": 5,
        "top_p": 0.8,
        "temperature": 0.7,
        "subtalker_dosample": True,
        "subtalker_top_k": 5,
        "subtalker_top_p": 0.8,
        "subtalker_temperature": 0.7,
        "max_new_tokens": 384,
        "non_streaming_mode": False,
    },
    "balanced": {
        "do_sample": True,
        "top_k": 20,
        "top_p": 0.8,
        "temperature": 0.8,
        "subtalker_dosample": True,
        "subtalker_top_k": 20,
        "subtalker_top_p": 0.8,
        "subtalker_temperature": 0.8,
        "max_new_tokens": 512,
        "non_streaming_mode": False,
    },
    "quality": {},
}


class EmotionalSpeaker:
    """Loads Qwen3-TTS Base once and caches per-(language, emotion) clone prompts."""

    def __init__(
        self,
        personality: str = DEFAULT_PERSONALITY,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
        ref_dir: Optional[Path] = None,
        ref_text: Optional[str] = None,
        ref_text_en: Optional[str] = None,
        ref_text_es: Optional[str] = None,
        clone_model_id: str = DEFAULT_CLONE_MODEL_ID,
        generation_options: Optional[Dict[str, Any]] = None,
        prompt_mode: str = "icl",
    ) -> None:
        self._personality = personality
        self._output_root = Path(output_root)
        self._ref_dir = Path(ref_dir) if ref_dir else self._output_root / personality / REF_SUBDIR
        self._runtime_dir = self._output_root / personality / RUNTIME_SUBDIR
        self._clone_model_id = clone_model_id
        self._generation_options = dict(generation_options or {})
        self._x_vector_only_mode = self._normalize_prompt_mode(prompt_mode) == "x_vector"
        if not self._ref_dir.is_dir():
            raise FileNotFoundError(f"Reference folder not found: {self._ref_dir}")

        self._ref_text_by_lang = self._build_ref_text_by_lang(
            ref_text=ref_text,
            ref_text_en=ref_text_en,
            ref_text_es=ref_text_es,
        )
        self._clone = Qwen3TTSModel.from_pretrained(
            self._clone_model_id,
            device_map=DEVICE_MAP,
            dtype=DTYPE,
            attn_implementation=ATTN_IMPL,
        )
        if hasattr(self._clone, "model"):
            self._clone.model.eval()
        self._prompt_cache: Dict[Tuple[str, str], object] = {}

    def say(
        self,
        text: str,
        language: str,
        emotion: str,
        output: Optional[str] = None,
    ) -> str:
        path, _ = self.synthesize(
            text=text,
            language=language,
            emotion=emotion,
            output=output,
        )
        return path

    def synthesize(
        self,
        text: str,
        language: str,
        emotion: str,
        output: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        paths, generation_options = self.synthesize_batch(
            texts=[text],
            language=language,
            emotion=emotion,
            output=output,
        )
        return paths[0], generation_options

    def synthesize_batch(
        self,
        texts: List[str],
        language: str,
        emotion: str | List[str],
        output: Optional[str] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        texts = self._normalize_texts(texts)
        lang = self._normalize_lang(language)
        emotions = self._normalize_emotions(emotion, len(texts))

        prompts = []
        for item_emotion in emotions:
            prompt_items = self._get_clone_prompt(lang, item_emotion)
            if len(prompt_items) != 1:
                raise ValueError(f"Expected one clone prompt for {lang}:{item_emotion}, got {len(prompt_items)}")
            prompts.append(prompt_items[0])
        generation_options = self._generation_options_for_texts(texts)
        non_streaming_mode = generation_options.pop("non_streaming_mode", False)
        with torch.inference_mode():
            wavs, sample_rate = self._clone.generate_voice_clone(
                text=texts,
                language=[lang] * len(texts),
                voice_clone_prompt=prompts,
                non_streaming_mode=non_streaming_mode,
                **generation_options,
            )
        out_paths = self._resolve_outputs(output, lang, emotions)
        if len(wavs) != len(out_paths):
            raise ValueError(f"Batch output mismatch: wavs={len(wavs)}, paths={len(out_paths)}")

        output_items = []
        total_trim_elapsed = 0.0
        total_trim_start = 0.0
        total_trim_end = 0.0
        trimmed_wavs = []
        for index, (out_path, wav) in enumerate(zip(out_paths, wavs), start=1):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            trimmed_wav, trim_info = trim_silence(wav, sample_rate, CLONED_AUDIO_TRIM_CONFIG)
            trimmed_wavs.append(trimmed_wav)
            self._write_wav_atomic(out_path, trimmed_wav, sample_rate)
            total_trim_elapsed += trim_info["trim_elapsed_seconds"]
            total_trim_start += trim_info["trim_removed_start_seconds"]
            total_trim_end += trim_info["trim_removed_end_seconds"]
            output_items.append(
                {
                    "index": index,
                    "output": str(out_path),
                    "emotion": emotions[index - 1],
                    "trim_elapsed_seconds": trim_info["trim_elapsed_seconds"],
                    "trim_removed_start_seconds": trim_info["trim_removed_start_seconds"],
                    "trim_removed_end_seconds": trim_info["trim_removed_end_seconds"],
                }
            )

        combined_path = None
        if len(trimmed_wavs) > 1:
            combined_path = self._resolve_combined_output(output, lang, emotions)
            combined_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_wav_atomic(
                combined_path,
                self._concat_with_pause(trimmed_wavs, sample_rate),
                sample_rate,
            )

        effective_generation_options = dict(generation_options)
        effective_generation_options["non_streaming_mode"] = non_streaming_mode
        effective_generation_options["batch_size"] = len(texts)
        effective_generation_options["emotions"] = emotions
        effective_generation_options["items"] = output_items
        effective_generation_options["combined_output"] = str(combined_path) if combined_path else None
        effective_generation_options["trim_elapsed_seconds"] = total_trim_elapsed
        effective_generation_options["trim_removed_start_seconds"] = total_trim_start
        effective_generation_options["trim_removed_end_seconds"] = total_trim_end
        return [str(path) for path in out_paths], effective_generation_options

    def preload(self, language: str, emotion: str) -> None:
        # Eagerly build and cache a clone prompt to remove first-turn latency.
        self._get_clone_prompt(self._normalize_lang(language), self._normalize_emotion(emotion))

    def preload_all(self) -> None:
        for language in LANG_TO_CODE:
            for emotion in sorted(PERSONALITY_TRAITS_BY_EMOTION):
                self.preload(language, emotion)

    def warmup(
        self,
        language: str = "Spanish",
        emotion: str = "neutral",
        text: str = "Hola.",
    ) -> None:
        lang = self._normalize_lang(language)
        emotion = self._normalize_emotion(emotion)
        prompt = self._get_clone_prompt(lang, emotion)
        generation_options = dict(self._generation_options)
        non_streaming_mode = generation_options.pop("non_streaming_mode", False)
        with torch.inference_mode():
            self._clone.generate_voice_clone(
                text=text,
                language=lang,
                voice_clone_prompt=prompt,
                non_streaming_mode=non_streaming_mode,
                **generation_options,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _resolve_output(self, output: Optional[str], language: str, emotion: str) -> Path:
        return self._resolve_outputs(output, language, [emotion])[0]

    def _resolve_outputs(
        self,
        output: Optional[str],
        language: str,
        emotions: List[str],
    ) -> List[Path]:
        code = LANG_TO_CODE[language]
        count = len(emotions)
        runtime_names = [
            f"voice_runtime_{code}_{emotions[index - 1]}.wav"
            if count == 1
            else f"voice_runtime_{code}_{emotions[index - 1]}_{index}.wav"
            for index in range(1, count + 1)
        ]
        if output:
            out_path = Path(output)
            if out_path.is_absolute():
                if count == 1:
                    return [out_path]
                if out_path.suffix:
                    return [
                        out_path.with_name(f"{out_path.stem}_{index}{out_path.suffix}")
                        for index in range(1, count + 1)
                    ]
                return [out_path / name for name in runtime_names]
            return [self._runtime_dir / out_path.parent / name for name in runtime_names]
        return [self._runtime_dir / name for name in runtime_names]

    def _resolve_combined_output(
        self,
        output: Optional[str],
        language: str,
        emotions: List[str],
    ) -> Path:
        code = LANG_TO_CODE[language]
        combined_name = f"voice_runtime_{code}_{'_'.join(emotions)}.wav"
        if output:
            out_path = Path(output)
            if out_path.is_absolute():
                if out_path.suffix:
                    return out_path.with_name(f"{out_path.stem}_combined{out_path.suffix}")
                return out_path / combined_name
            return self._runtime_dir / out_path.parent / combined_name
        return self._runtime_dir / combined_name

    @staticmethod
    def _concat_with_pause(wavs: List[np.ndarray], sample_rate: int) -> np.ndarray:
        pause_length = int(sample_rate * COMBINED_OUTPUT_PAUSE_MS / 1000)
        if pause_length <= 0 or len(wavs) <= 1:
            return np.concatenate(wavs)

        pause_shape = (pause_length, *wavs[0].shape[1:])
        pause = np.zeros(pause_shape, dtype=wavs[0].dtype)
        chunks = []
        for index, wav in enumerate(wavs):
            if index:
                chunks.append(pause)
            chunks.append(wav)
        return np.concatenate(chunks)

    def _generation_options_for_text(self, text: str) -> Dict[str, Any]:
        return self._generation_options_for_texts([text])

    def _generation_options_for_texts(self, texts: List[str]) -> Dict[str, Any]:
        generation_options = dict(self._generation_options)
        current_max_new_tokens = generation_options.get("max_new_tokens")
        if current_max_new_tokens is not None:
            required_max_new_tokens = max(
                self._min_max_new_tokens_for_text(text)
                for text in texts
            )
            if current_max_new_tokens < required_max_new_tokens:
                print(
                    (
                        "Increasing max_new_tokens from "
                        f"{current_max_new_tokens} to {required_max_new_tokens} "
                        "for this text length to avoid truncated speech."
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                generation_options["max_new_tokens"] = required_max_new_tokens
        return generation_options

    @staticmethod
    def _min_max_new_tokens_for_text(text: str) -> int:
        text_length = len(text.strip())
        estimated = max(MIN_MAX_NEW_TOKENS, text_length * MIN_MAX_NEW_TOKENS_PER_CHAR)
        return min(estimated, MAX_DYNAMIC_MAX_NEW_TOKENS)

    @staticmethod
    def _normalize_texts(texts: List[str]) -> List[str]:
        if not isinstance(texts, list):
            raise ValueError("Use 'texts' as a JSON list of up to 3 strings.")
        if not texts:
            raise ValueError("'texts' cannot be empty.")
        if len(texts) > MAX_BATCH_TEXTS:
            raise ValueError(f"'texts' supports up to {MAX_BATCH_TEXTS} phrases per request.")

        clean_texts = []
        for index, text in enumerate(texts, start=1):
            if not isinstance(text, str):
                raise ValueError(f"texts[{index - 1}] must be a string.")
            clean_text = text.strip()
            if not clean_text:
                raise ValueError(f"texts[{index - 1}] cannot be empty.")
            clean_texts.append(clean_text)
        return clean_texts

    @staticmethod
    def _write_wav_atomic(out_path: Path, wav: np.ndarray, sample_rate: int) -> None:
        tmp_path = out_path.with_name(f".{out_path.stem}.{time.time_ns()}.tmp{out_path.suffix}")
        try:
            sf.write(str(tmp_path), wav, sample_rate)
            tmp_path.replace(out_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _get_clone_prompt(self, language: str, emotion: str):
        key = (language, emotion)
        cached = self._prompt_cache.get(key)
        if cached is not None:
            return cached

        ref_wav, sample_rate = self._read_ref_audio(language, emotion)
        ref_text = None if self._x_vector_only_mode else self._ref_text_by_lang[language][emotion]
        with torch.inference_mode():
            prompt = self._clone.create_voice_clone_prompt(
                ref_audio=(ref_wav, sample_rate),
                ref_text=ref_text,
                x_vector_only_mode=self._x_vector_only_mode,
            )
        self._prompt_cache[key] = prompt
        return prompt

    def _read_ref_audio(self, language: str, emotion: str) -> Tuple[np.ndarray, int]:
        code = LANG_TO_CODE[language]
        ref_paths = [
            self._ref_dir / f"{prefix}_{code}_{emotion}.wav"
            for prefix in REF_FILE_PREFIXES
        ]
        ref_path = next((path for path in ref_paths if path.is_file()), None)
        if ref_path is None:
            expected = "\n".join(str(path) for path in ref_paths)
            raise FileNotFoundError(f"Reference audio not found. Tried:\n{expected}")
        wav, sample_rate = sf.read(str(ref_path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav, sample_rate

    @staticmethod
    def _build_ref_text_by_lang(
        ref_text: Optional[str],
        ref_text_en: Optional[str],
        ref_text_es: Optional[str],
    ) -> Dict[str, Dict[str, str]]:
        ref_text_by_lang = {
            language: dict(text_by_emotion)
            for language, text_by_emotion in DEFAULT_REF_TEXT_BY_LANG.items()
        }
        english_text = ref_text_en or ref_text
        spanish_text = ref_text_es or ref_text
        if english_text:
            ref_text_by_lang["English"] = {
                emotion: english_text for emotion in PERSONALITY_TRAITS_BY_EMOTION
            }
        if spanish_text:
            ref_text_by_lang["Spanish"] = {
                emotion: spanish_text for emotion in PERSONALITY_TRAITS_BY_EMOTION
            }
        return ref_text_by_lang

    @staticmethod
    def _normalize_lang(language: str) -> str:
        key = language.strip().lower()
        canonical = LANG_NAMES.get(key)
        if canonical is None:
            raise ValueError(f"Unsupported language '{language}'. Use 'English' or 'Spanish'.")
        return canonical

    @staticmethod
    def _normalize_emotion(emotion: str) -> str:
        key = emotion.strip().lower()
        if key not in PERSONALITY_TRAITS_BY_EMOTION:
            allowed = sorted(PERSONALITY_TRAITS_BY_EMOTION)
            raise ValueError(f"Unknown emotion '{emotion}'. Allowed: {allowed}")
        return key

    @classmethod
    def _normalize_emotions(cls, emotion: str | List[str], count: int) -> List[str]:
        if isinstance(emotion, list):
            if len(emotion) != count:
                raise ValueError(f"'emotion' list length must match texts length: {len(emotion)} != {count}")
            return [cls._normalize_emotion(item_emotion) for item_emotion in emotion]
        return [cls._normalize_emotion(emotion)] * count

    @staticmethod
    def _normalize_prompt_mode(prompt_mode: str) -> str:
        key = prompt_mode.strip().lower().replace("-", "_")
        if key not in {"icl", "x_vector"}:
            raise ValueError("Unsupported prompt mode. Use 'icl' or 'x_vector'.")
        return key


def _serve(speaker: EmotionalSpeaker, startup_config: Dict[str, Any]) -> None:
    # Line-delimited JSON protocol: one request per stdin line, one reply per stdout line.
    print(
        json.dumps(
            {
                "ready": True,
                "startup_seconds": startup_config.get("startup_seconds"),
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            started_at = time.perf_counter()
            is_batch = "texts" in req
            if is_batch:
                paths, generation_options = speaker.synthesize_batch(
                    texts=req["texts"],
                    language=req["language"],
                    emotion=req["emotion"],
                    output=req.get("output"),
                )
            else:
                path, generation_options = speaker.synthesize(
                    text=req["text"],
                    language=req["language"],
                    emotion=req["emotion"],
                    output=req.get("output"),
                )
                paths = [path]
            elapsed_seconds = time.perf_counter() - started_at
            response = _build_success_response(
                paths=paths,
                generation_options=generation_options,
                elapsed_seconds=elapsed_seconds,
                is_batch=is_batch,
            )
            print(json.dumps(response), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)


def _build_success_response(
    paths: List[str],
    generation_options: Dict[str, Any],
    elapsed_seconds: float,
    is_batch: bool,
) -> Dict[str, Any]:
    response = {
        "ok": True,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "effective_max_new_tokens": generation_options.get("max_new_tokens"),
        "batch_size": generation_options.get("batch_size", len(paths)),
        "trim_removed_start_seconds": round(
            generation_options.get("trim_removed_start_seconds", 0.0),
            3,
        ),
        "trim_removed_end_seconds": round(
            generation_options.get("trim_removed_end_seconds", 0.0),
            3,
        ),
    }
    if is_batch:
        response["outputs"] = [_response_filename(path) for path in paths]
        combined_output = generation_options.get("combined_output")
        if combined_output:
            response["combined_output"] = _response_filename(combined_output)
        response["items"] = [
            {
                "index": item["index"],
                "output": _response_filename(item["output"]),
                "emotion": item["emotion"],
                "trim_removed_start_seconds": round(item["trim_removed_start_seconds"], 3),
                "trim_removed_end_seconds": round(item["trim_removed_end_seconds"], 3),
            }
            for item in generation_options.get("items", [])
        ]
    else:
        response["output"] = _response_filename(paths[0])
    return response


def _response_filename(path: str) -> str:
    return Path(path).name


def _iter_preload_specs(raw_specs):
    for raw_spec in raw_specs:
        for item in raw_spec.split(","):
            spec = item.strip()
            if not spec:
                continue
            if ":" not in spec:
                raise ValueError(
                    f"Invalid --preload value '{spec}'. Use language:emotion, e.g. Spanish:happiness."
                )
            language, emotion = (part.strip() for part in spec.split(":", 1))
            yield language, emotion


def _prepare_low_latency_runtime(speaker: EmotionalSpeaker, args: argparse.Namespace) -> None:
    if args.preload_all:
        print("Preloading all language/emotion clone prompts...", file=sys.stderr, flush=True)
        speaker.preload_all()

    for language, emotion in _iter_preload_specs(args.preload):
        print(f"Preloading {language}:{emotion} clone prompt...", file=sys.stderr, flush=True)
        speaker.preload(language, emotion)

    if args.warmup:
        print(
            f"Warming up generation with {args.warmup_language}:{args.warmup_emotion}...",
            file=sys.stderr,
            flush=True,
        )
        speaker.warmup(
            language=args.warmup_language,
            emotion=args.warmup_emotion,
            text=args.warmup_text,
        )


def _build_generation_options(args: argparse.Namespace) -> Dict[str, Any]:
    options = dict(LATENCY_PRESETS[args.latency_preset])
    overrides = {
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "subtalker_dosample": args.subtalker_dosample,
        "subtalker_top_k": args.subtalker_top_k,
        "subtalker_top_p": args.subtalker_top_p,
        "subtalker_temperature": args.subtalker_temperature,
        "max_new_tokens": args.max_new_tokens,
        "non_streaming_mode": args.non_streaming_mode,
    }
    for key, value in overrides.items():
        if value is not None:
            options[key] = value
    return options


def _resolve_clone_model_id(args: argparse.Namespace) -> str:
    return args.clone_model_id or CLONE_MODEL_IDS[args.model_size]


def _build_startup_config(
    args: argparse.Namespace,
    clone_model_id: str,
    generation_options: Dict[str, Any],
) -> Dict[str, Any]:
    ref_dir = Path(args.ref_dir) if args.ref_dir else DEFAULT_OUTPUT_ROOT / args.personality / REF_SUBDIR
    runtime_dir = DEFAULT_OUTPUT_ROOT / args.personality / RUNTIME_SUBDIR
    return {
        "serve": args.serve,
        "personality": args.personality,
        "ref_dir": str(ref_dir),
        "runtime_dir": str(runtime_dir),
        "model_size": args.model_size,
        "clone_model_id": clone_model_id,
        "device_map": DEVICE_MAP,
        "dtype": str(DTYPE).replace("torch.", ""),
        "attn_implementation": ATTN_IMPL,
        "latency_preset": args.latency_preset,
        "prompt_mode": args.prompt_mode,
        "generation_options": generation_options,
        "max_batch_texts": MAX_BATCH_TEXTS,
        "trim_cloned_audio": {
            "enabled": CLONED_AUDIO_TRIM_CONFIG.enabled,
            "threshold": CLONED_AUDIO_TRIM_CONFIG.threshold,
            "padding_ms": CLONED_AUDIO_TRIM_CONFIG.padding_ms,
        },
        "preload_all": args.preload_all,
        "preload": args.preload,
        "warmup": args.warmup,
        "warmup_language": args.warmup_language,
        "warmup_emotion": args.warmup_emotion,
        "warmup_text": args.warmup_text,
    }


def _print_startup_config(startup_config: Dict[str, Any]) -> None:
    print("Runtime startup config:", file=sys.stderr, flush=True)
    print(json.dumps(startup_config, indent=2, sort_keys=True), file=sys.stderr, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hot emotional TTS for per-turn robot speech.")
    parser.add_argument("--text", help="Text to synthesize (one-shot mode).")
    parser.add_argument("--language", help="English | Spanish | en | es (one-shot mode).")
    parser.add_argument("--emotion", help=f"One of: {sorted(PERSONALITY_TRAITS_BY_EMOTION)} (one-shot mode).")
    parser.add_argument("--output", help=f"Output wav path. Relative paths are written under artifacts/<personality>/{RUNTIME_SUBDIR}/. If omitted, writes to artifacts/<personality>/{RUNTIME_SUBDIR}/voice_runtime_<lang>_<emotion>.wav")
    parser.add_argument("--personality", default=DEFAULT_PERSONALITY, help=f"Personality folder under artifacts/ (default: {DEFAULT_PERSONALITY}).")
    parser.add_argument("--ref_dir", "--ref-dir", dest="ref_dir", help="Folder containing reference wav files. Defaults to artifacts/<personality>/ref/voice_design_clone_ref/.")
    parser.add_argument("--ref_text", "--ref-text", dest="ref_text", help="Transcript shared by all reference audios. Use --ref_text_en/--ref_text_es to override per language.")
    parser.add_argument("--ref_text_en", "--ref-text-en", dest="ref_text_en", help="Transcript shared by English reference audios.")
    parser.add_argument("--ref_text_es", "--ref-text-es", dest="ref_text_es", help="Transcript shared by Spanish reference audios.")
    parser.add_argument("--model_size", "--model-size", dest="model_size", choices=sorted(CLONE_MODEL_IDS), default=DEFAULT_MODEL_SIZE, help=f"Base clone model size for runtime. 0.6B is faster; 1.7B keeps the old higher-quality runtime behavior (default: {DEFAULT_MODEL_SIZE}).")
    parser.add_argument("--clone_model_id", "--clone-model-id", dest="clone_model_id", help="Full Hugging Face/local model id. Overrides --model_size.")
    parser.add_argument("--latency_preset", "--latency-preset", dest="latency_preset", choices=sorted(LATENCY_PRESETS), default=DEFAULT_LATENCY_PRESET, help="Generation preset. fast uses compact sampling and caps output length for dialogue latency; quality uses qwen_tts defaults.")
    parser.add_argument("--prompt_mode", "--prompt-mode", dest="prompt_mode", choices=("icl", "x_vector"), default="icl", help="Voice clone prompt mode. icl keeps reference speech codes for emotion/style; x_vector is faster but usually less expressive.")
    parser.add_argument("--max_new_tokens", "--max-new-tokens", dest="max_new_tokens", type=int)
    parser.add_argument("--top_k", "--top-k", dest="top_k", type=int)
    parser.add_argument("--top_p", "--top-p", dest="top_p", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--subtalker_top_k", "--subtalker-top-k", dest="subtalker_top_k", type=int)
    parser.add_argument("--subtalker_top_p", "--subtalker-top-p", dest="subtalker_top_p", type=float)
    parser.add_argument("--subtalker_temperature", "--subtalker-temperature", dest="subtalker_temperature", type=float)
    parser.add_argument("--do_sample", "--do-sample", dest="do_sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--subtalker_dosample", "--subtalker-dosample", dest="subtalker_dosample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--non_streaming_mode", "--non-streaming-mode", dest="non_streaming_mode", action=argparse.BooleanOptionalAction, default=None, help="Forwarded to qwen_tts generate_voice_clone; disabled by fast/balanced presets because enabling it can add long silent regions.")
    parser.add_argument("--serve", action="store_true", help="Stay loaded and read JSON requests from stdin (one per line).")
    parser.add_argument("--preload-all", action="store_true", help="Build and cache every English/Spanish emotion prompt before serving. Startup is slower, but the first real turn for each emotion is faster.")
    parser.add_argument("--preload", action="append", default=[], help="Build and cache selected prompts before serving. Use language:emotion, repeat the flag, or comma-separate values, e.g. Spanish:happiness,English:fear.")
    parser.add_argument("--warmup", action="store_true", help="Run one short generation before serving so CUDA/model kernels are hot.")
    parser.add_argument("--warmup_language", "--warmup-language", dest="warmup_language", default="Spanish", help="Language used for --warmup (default: Spanish).")
    parser.add_argument("--warmup_emotion", "--warmup-emotion", dest="warmup_emotion", default="neutral", help="Emotion used for --warmup (default: neutral).")
    parser.add_argument("--warmup_text", "--warmup-text", dest="warmup_text", default="Hola.", help="Short text synthesized and discarded by --warmup.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.serve:
        missing = [name for name in ("text", "language", "emotion") if not getattr(args, name)]
        if missing:
            sys.exit(f"Missing required args: {missing}. Use --serve for hot mode.")

    startup_started_at = time.perf_counter()
    clone_model_id = _resolve_clone_model_id(args)
    generation_options = _build_generation_options(args)
    startup_config = _build_startup_config(
        args=args,
        clone_model_id=clone_model_id,
        generation_options=generation_options,
    )
    _print_startup_config(startup_config)

    speaker = EmotionalSpeaker(
        personality=args.personality,
        ref_dir=Path(args.ref_dir) if args.ref_dir else None,
        ref_text=args.ref_text,
        ref_text_en=args.ref_text_en,
        ref_text_es=args.ref_text_es,
        clone_model_id=clone_model_id,
        generation_options=generation_options,
        prompt_mode=args.prompt_mode,
    )
    _prepare_low_latency_runtime(speaker, args)
    startup_config["startup_seconds"] = round(time.perf_counter() - startup_started_at, 3)

    if args.serve:
        _serve(speaker, startup_config)
        return

    print(speaker.say(args.text, args.language, args.emotion, args.output))


if __name__ == "__main__":
    main()

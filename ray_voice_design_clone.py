from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


# Global model/runtime configuration used by both generation stages.
@dataclass(frozen=True)
class ModelConfig:
    voice_design_model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    clone_model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base" # "Qwen3-TTS-12Hz-0.6B-Base"
    device_map: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    attn_implementation: str = "flash_attention_2"


# Data contract for emotion-conditioned generation.
@dataclass(frozen=True)
class GenerationData:
    # Defines the canonical order used when iterating over emotions.
    emotion_order: List[str]
    # Maps each emotion to the exact style instruction used to build references.
    ref_instruct_by_emotion: Dict[str, str]
    # Shared reference texts used to generate EN/ES reference audios.
    ref_text_en: str
    ref_text_es: str
    # Per-emotion target sentences for final cloned outputs.
    sentences_en_by_emotion: Dict[str, str]
    sentences_es_by_emotion: Dict[str, str]


class VoiceDesignClonePipeline:
    # Orchestrates end-to-end generation: reference creation, prompt building, and cloning.
    def __init__(self, config: ModelConfig, data: GenerationData) -> None:
        self.config = config
        self.data = data
        self.ref_dir = Path("ref")
        self.output_dir = Path("output")

    def run(self) -> None:
        # Prepare filesystem and fail fast if emotion maps are inconsistent.
        self._ensure_output_dirs()
        self._validate_emotion_maps()

        # Load models once to avoid repeated initialization overhead.
        design_model = self._load_voice_design_model()
        clone_model = self._load_clone_model()

        # Process each emotion independently to keep prompt/style alignment explicit.
        for emotion in self.data.emotion_order:
            emotion_tag = self._file_tag(emotion)
            instruct = self.data.ref_instruct_by_emotion[emotion]

            # Create emotion-specific reference audios in EN and ES with the same style instruction.
            ref_wav_en, sr = self._generate_reference_audio(
                model=design_model,
                text=self.data.ref_text_en,
                language="English",
                instruct=instruct,
                out_path=self.ref_dir / f"voice_design_reference_en_{emotion_tag}.wav",
            )
            ref_wav_es, _ = self._generate_reference_audio(
                model=design_model,
                text=self.data.ref_text_es,
                language="Spanish",
                instruct=instruct,
                out_path=self.ref_dir / f"voice_design_reference_es_{emotion_tag}.wav",
            )

            # Build clone prompts from the generated references.
            voice_clone_prompt_en = self._build_clone_prompt(
                clone_model=clone_model,
                ref_wav=ref_wav_en,
                sample_rate=sr,
                ref_text=self.data.ref_text_en,
            )
            voice_clone_prompt_es = self._build_clone_prompt(
                clone_model=clone_model,
                ref_wav=ref_wav_es,
                sample_rate=sr,
                ref_text=self.data.ref_text_es,
            )

            # Generate one final cloned sentence per language for the same emotion.
            self._generate_single_clone(
                clone_model=clone_model,
                text=self.data.sentences_en_by_emotion[emotion],
                language="English",
                voice_clone_prompt=voice_clone_prompt_en,
                out_file=self.output_dir / f"clone_en_{emotion_tag}.wav",
            )
            self._generate_single_clone(
                clone_model=clone_model,
                text=self.data.sentences_es_by_emotion[emotion],
                language="Spanish",
                voice_clone_prompt=voice_clone_prompt_es,
                out_file=self.output_dir / f"clone_es_{emotion_tag}.wav",
            )

    def _ensure_output_dirs(self) -> None:
        # Ensure output folders exist before any write operation.
        self.ref_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def _generate_reference_audio(
        self,
        model: Qwen3TTSModel,
        text: str,
        language: str,
        instruct: str,
        out_path: Path,
    ) -> Tuple[np.ndarray, int]:
        # Generate one reference waveform and persist it for traceability/reuse.
        wavs, sample_rate = model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
        )
        sf.write(str(out_path), wavs[0], sample_rate)
        return wavs[0], sample_rate

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
        if set(self.data.ref_instruct_by_emotion.keys()) != base:
            raise ValueError("ref_instruct_by_emotion keys must match emotion_order")
        if set(self.data.sentences_en_by_emotion.keys()) != base:
            raise ValueError("sentences_en_by_emotion keys must match emotion_order")
        if set(self.data.sentences_es_by_emotion.keys()) != base:
            raise ValueError("sentences_es_by_emotion keys must match emotion_order")

    @staticmethod
    def _file_tag(emotion: str) -> str:
        # Normalize emotion names for file-safe, stable output names.
        return emotion.strip().lower().replace(" ", "_")


def build_default_data() -> GenerationData:
    # Shared speaker identity descriptor used as the base for all emotion instructions.
    base_identity = (
        "Voice of a 30-year-old men, AI assistant"
    )
    # Default dataset with one instruction and one EN/ES sentence per emotion.
    return GenerationData(
        emotion_order=[
            "anger",
            "disgust",
            "fear",
            "happiness",
            "neutral",
            "sadness",
            "surprise",
        ],
        ref_instruct_by_emotion={
            "anger": f"{base_identity}, robot companion. Clear articulation. " +
                      "Speaks with an angry, stern, and frustrated tone. Sharp delivery, loud, and lacking patience.",
            "disgust": f"{base_identity}, robot companion. Clear articulation. " +
                        "Speaks with a disgusted, repulsed, and uncomfortable tone. Expressing strong aversion and dislike.",
            "fear": f"{base_identity}, robot companion. Clear articulation. " +
                    "Speaks with a terrified, panicked, and trembling tone. Breathless and stuttering delivery, expressing extreme fear, distress, and high anxiety.",
            "happiness": f"{base_identity}, friendly and warm robot companion. Clear articulation. " +
                         "Speaks with a very happy, excited, and cheerful tone. Bright, enthusiastic, and highly engaging.",
            "neutral": f"{base_identity}, helpful robot companion. Clear articulation. " +
                       "Speaks with a neutral, calm, and composed tone. Even pacing, informative, without strong emotion.",
            "sadness": f"{base_identity}, empathetic robot companion. Clear articulation. " +
                       "Speaks with a heartbroken, devastated, and tearful tone. Voice cracking with deep grief, very slow pacing, heavy sighs, expressing profound sorrow and despair.",
            "surprise": f"{base_identity}, expressive robot companion. Clear articulation. " +
                        "Speaks with a highly surprised, astonished, and amazed tone. Wide-eyed, breathless, and sudden energy.",
        },
        # Base reference texts used to build clone prompts in each language.
        ref_text_en=(
            "Hello. I think we have a solid plan to start our work today. I enjoy exploring new ideas, figuring out complex details, and simply being here to assist you whenever you need it."
        ),
        ref_text_es=(
            "Hola. Creo que tenemos un buen plan para comenzar nuestro trabajo hoy. Disfruto explorando nuevas ideas, resolviendo detalles complejos, y simplemente estando aquí para ayudarte cuando lo necesites."
        ),
        sentences_en_by_emotion={
            "anger": "It makes me really upset when people don't listen. We need to respect the rules!",
            "disgust": "Ugh, that smells like spoiled milk. We should throw it away immediately.",
            "fear": "Wait, did you hear that loud noise? I... I'm really scared! Please, don't leave me alone in the dark!",
            "happiness": "Congratulations! That's wonderful news! I'm so happy for you.",
            "neutral": "The schedule for today includes a walk in the park and reading a book.",
            "sadness": "That's disheartening to hear. I'm here to help you get through this.",
            "surprise": "Oh wow, I hadn't even thought about that! That is a truly unexpected and fantastic idea.",
        },
        sentences_es_by_emotion={
            "anger": "¡Me molesta mucho cuando la gente no escucha! Tenemos que respetar las reglas.",
            "disgust": "Uf, eso huele a leche en mal estado. Deberíamos tirarlo de inmediato.",
            "fear": "¡Espera! ¿Has oído ese ruido tan fuerte? Yo... ¡tengo mucho miedo! ¡Por favor, no me dejes solo en la oscuridad!",
            "happiness": "¡Felicidades! ¡Son noticias maravillosas! Me alegro mucho por ti.",
            "neutral": "El horario para hoy incluye un paseo por el parque y leer un libro.",
            "sadness": "Es desalentador escuchar eso. Estoy aquí para ayudarte a superar esto.",
            "surprise": "¡Oh, vaya, ni siquiera había pensado en eso! Es una idea realmente inesperada y fantástica.",
        },
    )


def main() -> None:
    # Entry point used when running this file as a script.
    pipeline = VoiceDesignClonePipeline(config=ModelConfig(), data=build_default_data())
    pipeline.run()


if __name__ == "__main__":
    main()
from typing import Dict, List, TypedDict


class GenerationDefaults(TypedDict):
    emotion_order: List[str]
    ref_instruct_by_emotion: Dict[str, str]
    voice_design_ref_text_en_by_emotion: Dict[str, str]
    voice_design_ref_text_es_by_emotion: Dict[str, str]
    voice_clone_refs_text_en_by_emotion: Dict[str, str]
    voice_clone_refs_text_es_by_emotion: Dict[str, str]
    sentences_en_by_emotion: Dict[str, str]
    sentences_es_by_emotion: Dict[str, str]
    personality_folder: str


# Folder label used to group outputs for this personality preset.
DEFAULT_PERSONALITY_FOLDER = "personality_1"


# Base identity descriptor shared by all emotion instructions.
BASE_IDENTITY = "Voice of a 30-year-old man, AI assistant"


# Emotion-specific personality/style traits used to build reference instructions.
PERSONALITY_TRAITS_BY_EMOTION: Dict[str, str] = {
    "anger": "robot companion. Clear articulation. Speaks with an angry, stern, and frustrated tone. Sharp delivery, loud, and lacking patience.",
    "disgust": "robot companion. Clear articulation. Speaks with a disgusted, repulsed, and uncomfortable tone. Expressing strong aversion and dislike.",
    "fear": "robot companion. Clear articulation. Speaks with a terrified, panicked, and trembling tone. Breathless and stuttering delivery, expressing extreme fear, distress, and high anxiety.",
    "happiness": "friendly and warm robot companion. Clear articulation. Speaks with a very happy, excited, and cheerful tone. Bright, enthusiastic, and highly engaging.",
    "neutral": "helpful robot companion. Clear articulation. Speaks with a neutral, calm, and composed tone. Even pacing, informative, without strong emotion.",
    "sadness": "empathetic robot companion. Clear articulation. Speaks with a heartbroken, devastated, and tearful tone. Voice cracking with deep grief, very slow pacing, heavy sighs, expressing profound sorrow and despair.",
    "surprise": "expressive robot companion. Clear articulation. Speaks with a highly surprised, astonished, and amazed tone. Wide-eyed, breathless, and sudden energy.",
}


def build_ref_instruct_by_emotion(base_identity: str = BASE_IDENTITY) -> Dict[str, str]:
    # Compose final per-emotion instructions from identity + personality traits.
    return {
        emotion: f"{base_identity}, {traits}"
        for emotion, traits in PERSONALITY_TRAITS_BY_EMOTION.items()
    }


# Canonical emotion order used across instruction/sentence maps.
DEFAULT_EMOTION_ORDER: List[str] = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]


# Base transcripts used as defaults for all emotions.
BASE_VOICE_DESIGN_REF_TEXT_EN = (
    "Hello. I think we have a solid plan to start our work today. I enjoy exploring new ideas, figuring out complex details, and simply being here to assist you whenever you need it."
)
BASE_VOICE_DESIGN_REF_TEXT_ES = (
    "Hola. Creo que tenemos un buen plan para comenzar nuestro trabajo hoy. Disfruto explorando nuevas ideas, resolviendo detalles complejos, y simplemente estando aquí para ayudarte cuando lo necesites."
)

# Per-emotion transcripts used by voice_design_clone.py to generate refs and build clone prompts.
VOICE_DESIGN_REF_TEXT_EN_BY_EMOTION: Dict[str, str] = {
    emotion: BASE_VOICE_DESIGN_REF_TEXT_EN
    for emotion in DEFAULT_EMOTION_ORDER
}
VOICE_DESIGN_REF_TEXT_ES_BY_EMOTION: Dict[str, str] = {
    emotion: BASE_VOICE_DESIGN_REF_TEXT_ES
    for emotion in DEFAULT_EMOTION_ORDER
}


# Per-emotion transcripts used by voice_clone.py for uploaded refs and clone prompts.
# Fill these with the exact transcript of each uploaded audio file:
# - voice_clone_ref_en_<emotion>.wav
# - voice_clone_ref_es_<emotion>.wav
VOICE_CLONE_REFS_TEXT_EN_BY_EMOTION: Dict[str, str] = {
    # Replace with the exact transcript for voice_clone_ref_en_anger.wav
    "anger": "Hello. I think we have a solid plan to start our work today. I enjoy exploring new ideas, figuring out complex details, and simply being here to assist you whenever you need it.",
    # Replace with the exact transcript for voice_clone_ref_en_disgust.wav
    "disgust": "Hello. I think we have a solid plan to start our work today. I enjoy exploring new ideas, figuring out complex details, and simply being here to assist you whenever you need it.",
    # Replace with the exact transcript for voice_clone_ref_en_fear.wav
    "fear": "Hello. I think we have a solid plan to start our work today. I enjoy exploring new ideas, figuring out complex details, and simply being here to assist you whenever you need it.",
    # Replace with the exact transcript for voice_clone_ref_en_happiness.wav
    "happiness": "Hello. I think we have a solid plan to start our work today. I enjoy exploring new ideas, figuring out complex details, and simply being here to assist you whenever you need it.",
    # Replace with the exact transcript for voice_clone_ref_en_neutral.wav
    "neutral": "Hello. I think we have a solid plan to start our work today. I enjoy exploring new ideas, figuring out complex details, and simply being here to assist you whenever you need it.",
    # Replace with the exact transcript for voice_clone_ref_en_sadness.wav
    "sadness": "Hello. I think we have a solid plan to start our work today. I enjoy exploring new ideas, figuring out complex details, and simply being here to assist you whenever you need it.",
    # Replace with the exact transcript for voice_clone_ref_en_surprise.wav
    "surprise": "Hello. I think we have a solid plan to start our work today. I enjoy exploring new ideas, figuring out complex details, and simply being here to assist you whenever you need it.",
}
VOICE_CLONE_REFS_TEXT_ES_BY_EMOTION: Dict[str, str] = {
    # Replace with the exact transcript for voice_clone_ref_es_anger.wav
    "anger": "Hola. Creo que tenemos un buen plan para comenzar nuestro trabajo hoy. Disfruto explorando nuevas ideas, resolviendo detalles complejos, y simplemente estando aquí para ayudarte cuando lo necesites.",
    # Replace with the exact transcript for voice_clone_ref_es_disgust.wav
    "disgust": "Hola. Creo que tenemos un buen plan para comenzar nuestro trabajo hoy. Disfruto explorando nuevas ideas, resolviendo detalles complejos, y simplemente estando aquí para ayudarte cuando lo necesites.",
    # Replace with the exact transcript for voice_clone_ref_es_fear.wav
    "fear": "Hola. Creo que tenemos un buen plan para comenzar nuestro trabajo hoy. Disfruto explorando nuevas ideas, resolviendo detalles complejos, y simplemente estando aquí para ayudarte cuando lo necesites.",
    # Replace with the exact transcript for voice_clone_ref_es_happiness.wav
    "happiness": "Hola. Creo que tenemos un buen plan para comenzar nuestro trabajo hoy. Disfruto explorando nuevas ideas, resolviendo detalles complejos, y simplemente estando aquí para ayudarte cuando lo necesites.",
    # Replace with the exact transcript for voice_clone_ref_es_neutral.wav
    "neutral": "Hola. Creo que tenemos un buen plan para comenzar nuestro trabajo hoy. Disfruto explorando nuevas ideas, resolviendo detalles complejos, y simplemente estando aquí para ayudarte cuando lo necesites.",
    # Replace with the exact transcript for voice_clone_ref_es_sadness.wav
    "sadness": "Hola. Creo que tenemos un buen plan para comenzar nuestro trabajo hoy. Disfruto explorando nuevas ideas, resolviendo detalles complejos, y simplemente estando aquí para ayudarte cuando lo necesites.",
    # Replace with the exact transcript for voice_clone_ref_es_surprise.wav
    "surprise": "Hola. Creo que tenemos un buen plan para comenzar nuestro trabajo hoy. Disfruto explorando nuevas ideas, resolviendo detalles complejos, y simplemente estando aquí para ayudarte cuando lo necesites.",
}


# Per-emotion final target sentences for cloned generation.
DEFAULT_SENTENCES_EN_BY_EMOTION: Dict[str, str] = {
    "anger": "It makes me really upset when people don't listen. We need to respect the rules!",
    "disgust": "Ugh, that smells like spoiled milk. We should throw it away immediately.",
    "fear": "Wait, did you hear that loud noise? I... I'm really scared! Please, don't leave me alone in the dark!",
    "happiness": "Congratulations! That's wonderful news! I'm so happy for you.",
    "neutral": "The schedule for today includes a walk in the park and reading a book.",
    "sadness": "That's disheartening to hear. I'm here to help you get through this.",
    "surprise": "Oh wow, I hadn't even thought about that! That is a truly unexpected and fantastic idea.",
}

DEFAULT_SENTENCES_ES_BY_EMOTION: Dict[str, str] = {
    "anger": "¡Me molesta mucho cuando la gente no escucha! Tenemos que respetar las reglas.",
    "disgust": "Uf, eso huele a leche en mal estado. Deberíamos tirarlo de inmediato.",
    "fear": "¡Espera! ¿Has oído ese ruido tan fuerte? Yo... ¡tengo mucho miedo! ¡Por favor, no me dejes solo en la oscuridad!",
    "happiness": "¡Felicidades! ¡Son noticias maravillosas! Me alegro mucho por ti.",
    "neutral": "El horario para hoy incluye un paseo por el parque y leer un libro.",
    "sadness": "Es desalentador escuchar eso. Estoy aquí para ayudarte a superar esto.",
    "surprise": "¡Oh, vaya, ni siquiera había pensado en eso! Es una idea realmente inesperada y fantástica.",
}


def build_default_generation_config() -> GenerationDefaults:
    # Keep all generation defaults centralized in one place.
    return {
        "emotion_order": DEFAULT_EMOTION_ORDER,
        "ref_instruct_by_emotion": build_ref_instruct_by_emotion(),
        "voice_design_ref_text_en_by_emotion": VOICE_DESIGN_REF_TEXT_EN_BY_EMOTION,
        "voice_design_ref_text_es_by_emotion": VOICE_DESIGN_REF_TEXT_ES_BY_EMOTION,
        "voice_clone_refs_text_en_by_emotion": VOICE_CLONE_REFS_TEXT_EN_BY_EMOTION,
        "voice_clone_refs_text_es_by_emotion": VOICE_CLONE_REFS_TEXT_ES_BY_EMOTION,
        "sentences_en_by_emotion": DEFAULT_SENTENCES_EN_BY_EMOTION,
        "sentences_es_by_emotion": DEFAULT_SENTENCES_ES_BY_EMOTION,
        "personality_folder": DEFAULT_PERSONALITY_FOLDER,
    }

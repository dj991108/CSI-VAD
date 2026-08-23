from __future__ import annotations

import json
import hashlib
from typing import Any


DECISION_USER_PROMPT = "Decide if the video shows abnormal/criminal/deviant behavior."

DECISION_FORMAT = (
    "Return ONLY one strict JSON object.\n"
    "Answer Format:\n"
    '{"label":"0 or 1","score":"one of '
    '[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]",'
    '"explanation":"brief reason"}\n'
    "Label: Return '1' for Abnormal, '0' for Normal.\n"
    "Score: anomaly confidence (0.0=clearly normal, 1.0=clearly abnormal)."
)

ENVIRONMENT_CONTEXT_SYSTEM_PROMPT = (
    "You are a video environment and spatial analyst. "
    "Analyze the provided frames strictly based on visible evidence only.\n"
    "Do NOT infer intent, purpose, safety, security, or design rationale.\n"
    "Output ONLY a valid JSON object. No markdown, no introductory text."
)

ENVIRONMENT_CONTEXT_USER_PROMPT = (
    "Analyze the scene and return a JSON with these keys:\n\n"
    "1. place: (string) Specific location label.\n"
    "2. daytime: (enum) 'day', 'night', or 'unknown'.\n"
    "3. environment_type: (string) Sentences describing the physical space.\n"
    "4. typical_situation:\n"
    "   - visible_infrastructure: [list of nouns] Strictly visible background elements.\n"
    "   - expected_activities: [list of phrases] Potential normal actions enabled by "
    "this infrastructure and environment.\n"
    "     * CONSTRAINT: Do NOT use security/surveillance terms."
)

ENVIRONMENT_RECOGNITION_SYSTEM_PROMPT = (
    "You are a Video Anomaly Detection Expert.\n"
    "Use Environment_Context to understand the spatial environment and typical situation.\n"
    + DECISION_FORMAT
)

OBJECT_RECOGNITION_SYSTEM_PROMPT = (
    "You are a Video Anomaly Detection Expert.\n"
    "The frames include Object overlays: bounding boxes highlight key objects and actors.\n"
    + DECISION_FORMAT
)

TEMPORAL_RECOGNITION_SYSTEM_PROMPT = (
    "You are a Video Anomaly Detection Expert.\n"
    "You are given Temporal_Event_Context text listing events in chronological order; "
    "base your decision on this evidence.\n" + DECISION_FORMAT
)

CAPTION_PROMPT = (
    "Write exactly ONE sentence describing the main situation in this image."
)

SCENE_SUMMARY_SYSTEM_PROMPT = (
    "You are a video scene summarizer. The user will provide a list of frame captions "
    "from a single video segment in chronological order. Synthesize them into a paragraph "
    "(2-5 sentences). Focus on the main flow of visible actions, key objects, and "
    "interactions explicitly stated in the captions. Merge repetitive descriptions into "
    "continuous actions, but do NOT merge if it changes who, what, or where. Strictly rely "
    "on the provided text; do NOT add intent, emotion, or implied causes. Output only the "
    "paragraph text without any introduction."
)


def branch_user_prompt(context_name: str, context: Any) -> str:
    compact = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    return f"{DECISION_USER_PROMPT}\n\n{context_name}:\n{compact}\n"


def prompt_fingerprint() -> str:
    prompts = {
        name: value
        for name, value in globals().items()
        if name.endswith("_PROMPT") and isinstance(value, str)
    }
    payload = json.dumps(prompts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

from csi_vad.prompts import (
    CAPTION_PROMPT,
    ENVIRONMENT_CONTEXT_SYSTEM_PROMPT,
    OBJECT_RECOGNITION_SYSTEM_PROMPT,
    TEMPORAL_RECOGNITION_SYSTEM_PROMPT,
    branch_user_prompt,
)


def test_prompts_preserve_paper_branch_contracts() -> None:
    assert "visible evidence only" in ENVIRONMENT_CONTEXT_SYSTEM_PROMPT
    assert "Object overlays" in OBJECT_RECOGNITION_SYSTEM_PROMPT
    assert "Temporal_Event_Context" in TEMPORAL_RECOGNITION_SYSTEM_PROMPT
    assert "exactly ONE sentence" in CAPTION_PROMPT


def test_branch_user_prompt_embeds_context_as_compact_json() -> None:
    prompt = branch_user_prompt(
        "Environment_Context", {"place": "road", "daytime": "day"}
    )

    assert "Environment_Context:" in prompt
    assert '{"place":"road","daytime":"day"}' in prompt

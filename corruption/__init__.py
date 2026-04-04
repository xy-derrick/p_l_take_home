from corruption.sync_shift import apply_sync_shift
from corruption.speaker_swap import apply_speaker_swap
from corruption.artifact_inject import inject_artifacts
from corruption.sfx_mistime import apply_sfx_mistime
from corruption.music_mood_swap import apply_music_mood_swap

CORRUPTION_FUNCTIONS = {
    "apply_sync_shift": apply_sync_shift,
    "apply_speaker_swap": apply_speaker_swap,
    "inject_artifacts": inject_artifacts,
    "apply_sfx_mistime": apply_sfx_mistime,
    "apply_music_mood_swap": apply_music_mood_swap,
}

"""VoxCeleb dataset loading utilities."""

from pathlib import Path


def load_voxceleb1_test(root: Path) -> list[tuple[Path, int]]:
    """Load VoxCeleb1 test set.

    Args:
        root: Path to VoxCeleb1 root directory

    Returns:
        List of (audio_path, speaker_id) pairs where speaker_id is an integer
        mapping from the speaker directory name (e.g., id10001 -> 0)

    The VoxCeleb1 directory structure is:
        voxceleb1/
            id10001/
                1zcIwhmdeo4/
                    00001.wav
                    00002.wav
            id10002/
                ...
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"VoxCeleb root not found: {root}")

    # Find all speaker directories (id####)
    speaker_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("id")])

    if not speaker_dirs:
        raise ValueError(f"No speaker directories found in {root}")

    # Create speaker ID mapping (speaker_dir_name -> integer ID)
    speaker_to_id = {d.name: i for i, d in enumerate(speaker_dirs)}

    # Collect all audio files with their speaker IDs
    audio_files = []
    for speaker_dir in speaker_dirs:
        speaker_id = speaker_to_id[speaker_dir.name]

        # Find all .wav files recursively under this speaker
        wav_files = list(speaker_dir.rglob("*.wav"))

        for wav_file in wav_files:
            audio_files.append((wav_file, speaker_id))

    if not audio_files:
        raise ValueError(f"No audio files found in {root}")

    return audio_files

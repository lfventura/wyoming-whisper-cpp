"""Utility for downloading models."""
import subprocess
from pathlib import Path
from typing import Union

# https://huggingface.co/ggerganov/whisper.cpp/tree/main
WHISPER_CPP_MODELS = [
    "tiny",
    "tiny-q8_0",
    "tiny-q5_1",
    "tiny.en",
    "tiny.en-q8_0",
    "tiny.en-q5_1",
    "base",
    "base-q8_0",
    "base-q5_1",
    "base.en",
    "base.en-q5_1",
    "base.en-q8_0",
    "small",
    "small-q8_0",
    "small-q5_1",
    "small.en",
    "small.en-q8_0",
    "small.en-q5_1",
    "medium",
    "medium-q8_0",
    "medium-q5_0",
    "medium.en",
    "medium.en-q8_0",
    "medium.en-q5_0",
    "large-v1",
    "large-v2",
    "large-v2-q8_0",
    "large-v2-q5_0",
    "large-v3",
    "large-v3-q5_0",
    "large-v3-turbo",
    "large-v3-turbo-q8_0",
    "large-v3-turbo-q5_0",
]


def model_name_to_path(model_name: str, dest_dir: Union[str, Path]) -> Path:
    return Path(dest_dir) / f"ggml-{model_name}.bin"


def download_model(
    whisper_cpp_dir: Union[str, Path], model_name: str, dest_dir: Union[str, Path]
) -> None:
    """Downloads whisper.cpp model using the ggml download script."""
    whisper_cpp_dir = Path(whisper_cpp_dir)
    dest_dir = Path(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)
    script_path = whisper_cpp_dir / "models" / "download-ggml-model.sh"
    subprocess.check_call([str(script_path), str(model_name), str(dest_dir)])

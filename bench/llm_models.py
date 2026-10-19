"""Build the summary-model GGUFs ourselves, from each publisher's own weights.

    bench/data/tools/venv-convert/bin/python -I bench/llm_models.py [name ...]

For every model: download the official safetensors at a pinned commit, convert them to
GGUF with llama.cpp's convert_hf_to_gguf.py, quantize to Q4_K_M with llama-quantize,
record the result in `bench/llm_models.json` (repo, commit, licence, size, sha256), and
delete the downloaded weights and the f16 intermediate. The pinned commit is enough to
rebuild the file; the sha256 is what a user's download will be checked against.

llama.cpp is the one the summary sidecar is built with (llama-cpp-sys-2 0.1.160 vendors
ggml-org/llama.cpp d81235049384534c167caea52b85a694f6103d14), checked out and built in
bench/data/tools/llama.cpp. Licences were read from each repository first
(bench/README.md); nothing here decides them.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
DATA = BENCH / "data"
TOOLS = DATA / "tools"
LLAMA = TOOLS / "llama.cpp"
SOURCES = DATA / "llm-src"  # downloaded weights, deleted after conversion
OUT = DATA / "models" / "llm"
MANIFEST = BENCH / "llm_models.json"
QUANT = "Q4_K_M"

# name: (repository, licence as read from the repository, device tier)
MODELS = {
    "kanana-1.5-2.1b": ("kakaocorp/kanana-1.5-2.1b-instruct-2505", "Apache-2.0", "8GB"),
    "midm-2.0-mini": ("K-intelligence/Midm-2.0-Mini-Instruct", "MIT", "8GB"),
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "Apache-2.0", "8GB"),
    "gemma-4-e4b": ("google/gemma-4-E4B-it", "Apache-2.0", "8GB"),
    "ax-4.0-light": ("skt/A.X-4.0-Light", "Apache-2.0 (ships a NOTICE: built with Qwen 2.5)", "16GB"),
    "kanana-1.5-8b": ("kakaocorp/kanana-1.5-8b-instruct-2505", "Apache-2.0", "16GB"),
    "qwen3.5-9b": ("Qwen/Qwen3.5-9B", "Apache-2.0", "16GB"),
    "midm-2.0-base": ("K-intelligence/Midm-2.0-Base-Instruct", "MIT", "16GB"),
}

# what conversion reads; everything else in a repository (other formats, images) is skipped
# (no *.py: a repository's own code is never run, and conversion does not need it)
WANTED = ["*.json", "*.safetensors", "tokenizer*", "*.model", "*.tiktoken", "*.jinja", "*.txt"]


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def transformers_version() -> str:
    import transformers

    return transformers.__version__


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def build(name: str) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    repo, licence, tier = MODELS[name]
    commit = HfApi().model_info(repo).sha
    src = SOURCES / name
    log(f"{name}: downloading {repo}@{commit[:12]}")
    snapshot_download(repo, revision=commit, local_dir=src, allow_patterns=WANTED)

    OUT.mkdir(parents=True, exist_ok=True)
    f16 = OUT / f"{name}-f16.gguf"
    gguf = OUT / f"{name}-{QUANT}.gguf"
    log(f"{name}: converting")
    # not -I: the converter imports its own package from beside it (pinned llama.cpp);
    # the downloaded weights are only ever an argument, never on the path
    subprocess.run(
        [sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(src), "--outtype", "f16", "--outfile", str(f16)],
        check=True,
        cwd=LLAMA,
    )
    log(f"{name}: quantizing to {QUANT}")
    subprocess.run([str(LLAMA / "build/bin/llama-quantize"), str(f16), str(gguf), QUANT], check=True,
                   stdout=subprocess.DEVNULL)

    entry = {
        "repo": repo,
        "commit": commit,
        "licence": licence,
        "tier": tier,
        "file": gguf.name,
        "quant": QUANT,
        "bytes": gguf.stat().st_size,
        "sha256": sha256(gguf),
        "llama_cpp": "ggml-org/llama.cpp@d81235049384534c167caea52b85a694f6103d14",
        # the converter reads the model config through transformers; llama.cpp pins
        # 4.57.6, which predates some architectures (Gemma 4 needed a newer one)
        "transformers": transformers_version(),
    }
    f16.unlink()
    shutil.rmtree(src)
    log(f"{name}: {entry['bytes'] / 2**30:.2f} GiB, sha256 {entry['sha256'][:12]}…; weights deleted")
    return entry


def main() -> None:
    names = sys.argv[1:] or list(MODELS)
    manifest = load_manifest()
    for name in names:
        if name in manifest and (OUT / manifest[name]["file"]).exists():
            log(f"{name}: already built")
            continue
        try:
            manifest[name] = build(name)
        except Exception as exc:  # noqa: BLE001 - report and go on to the next model
            log(f"{name}: FAILED: {exc}")
            continue
        MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

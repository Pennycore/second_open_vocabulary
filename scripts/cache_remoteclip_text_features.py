from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ov_probe.io import load_config  # noqa: E402
from ov_probe.prompts import build_prompt_bank  # noqa: E402


def _all_unique_prompts(bank: dict) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    names = list(bank["class_names"]) + list(bank["distractors"])
    for group in ("A", "B"):
        for name in names:
            for prompt in bank["groups"][group][name]:
                if prompt not in seen:
                    seen.add(prompt)
                    result.append(prompt)
    return result


def _remote_program(prompts: list[str], model_name: str, checkpoint: str, batch_size: int) -> str:
    prompts_json = json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    prompts_b64 = base64.b64encode(prompts_json).decode("ascii")
    return f'''import base64, hashlib, json, sys
import numpy as np
import open_clip
import torch

prompts = json.loads(base64.b64decode({prompts_b64!r}).decode("utf-8"))
model_name = {model_name!r}
checkpoint = {checkpoint!r}
model = open_clip.create_model(model_name, pretrained=None)
state = torch.load(checkpoint, map_location="cpu", weights_only=True)
state = state.get("state_dict", state)
state = {{key.removeprefix("module."): value for key, value in state.items()}}
missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    raise RuntimeError(f"checkpoint mismatch: missing={{missing[:5]}}, unexpected={{unexpected[:5]}}")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval().to(device)
tokenizer = open_clip.get_tokenizer(model_name)
parts = []
with torch.inference_mode():
    for start in range(0, len(prompts), {batch_size}):
        tokens = tokenizer(prompts[start:start + {batch_size}]).to(device)
        part = model.encode_text(tokens).float()
        part = part / part.norm(dim=-1, keepdim=True)
        parts.append(part.cpu())
features = torch.cat(parts).numpy().astype("<f4", copy=False)
digest = hashlib.sha256()
with open(checkpoint, "rb") as handle:
    while True:
        chunk = handle.read(8 * 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
payload = {{
    "model_name": model_name,
    "checkpoint": checkpoint,
    "checkpoint_sha256": digest.hexdigest(),
    "python": sys.version,
    "torch": torch.__version__,
    "open_clip_file": open_clip.__file__,
    "device": device,
    "shape": list(features.shape),
    "features_base64": base64.b64encode(features.tobytes()).decode("ascii"),
}}
print(json.dumps(payload, separators=(",", ":")))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache RemoteCLIP text features via read-only remote inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--identity-file", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, PROJECT_ROOT)
    output = Path(cfg["paths"]["text_feature_cache"])
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite text cache: {output} / {metadata_path}")
    text_cfg = cfg.get("text_encoding", {})
    host = str(text_cfg["remote_host"])
    remote_python = str(text_cfg["remote_python"])
    remote_checkpoint = str(text_cfg["remote_checkpoint"])
    prompts = _all_unique_prompts(build_prompt_bank(cfg))
    program = _remote_program(
        prompts,
        str(cfg["model"]["model_name"]),
        remote_checkpoint,
        int(cfg["model"].get("text_batch_size", 128)),
    )
    program_b64 = base64.b64encode(program.encode("utf-8")).decode("ascii")
    remote_command = f'''{remote_python} -c "import base64;exec(base64.b64decode('{program_b64}'))"'''
    completed = subprocess.run(
        [
            "ssh",
            "-i",
            str(Path(args.identity_file).expanduser()),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            remote_command,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Remote text encoding failed ({completed.returncode}): {completed.stderr[-2000:]}")
    lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"Remote encoder returned no JSON payload: {completed.stdout[-2000:]}")
    payload = json.loads(lines[-1])
    expected_shape = (len(prompts), int(cfg["model"]["feature_dim"]))
    if tuple(payload["shape"]) != expected_shape:
        raise RuntimeError(f"Remote text feature shape mismatch: {payload['shape']} != {expected_shape}")
    raw = base64.b64decode(payload.pop("features_base64"))
    features = np.frombuffer(raw, dtype="<f4").reshape(expected_shape).copy()
    if not np.isfinite(features).all():
        raise RuntimeError("Remote text features contain non-finite values.")
    norms = np.linalg.norm(features, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-5):
        raise RuntimeError("Remote text features are not L2-normalized.")
    prompt_bytes = json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    metadata = {
        **payload,
        "cache_format_version": 1,
        "feature_dimension": expected_shape[1],
        "prompt_count": len(prompts),
        "prompts_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "remote_access": "read-only inference via stdout; no remote file created",
        "scientific_role": "fixed RemoteCLIP text embeddings for Stage 0",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(
            handle,
            format_version=np.asarray([1], dtype=np.int16),
            prompts=np.asarray(prompts),
            features=features.astype(np.float32),
        )
    with metadata_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"output": str(output), "shape": list(features.shape), "checkpoint_sha256": metadata["checkpoint_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

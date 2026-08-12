from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class InputValidationError(ValueError):
    """Raised when an external input violates the registered protocol."""


@dataclass
class FeatureBundle:
    features: np.ndarray
    class_names: list[str]
    metadata: dict[str, Any]
    sample_counts: np.ndarray | None = None
    prototype_ids: list[str] | None = None
    cluster_sizes: np.ndarray | None = None
    cam_labels: list[str] | None = None
    sam3_source_labels: list[str] | None = None


def load_config(path: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise InputValidationError("Configuration root must be a mapping.")
    root = Path(project_root).resolve() if project_root else config_path.parent.parent.resolve()
    cfg["_meta"] = {"config_path": str(config_path), "project_root": str(root)}
    _validate_protocol(cfg)
    for key, value in list(cfg.get("paths", {}).items()):
        if value is None:
            continue
        candidate = Path(os.path.expandvars(os.path.expanduser(str(value))))
        if not candidate.is_absolute():
            candidate = root / candidate
        cfg["paths"][key] = str(candidate.resolve())
    _validate_path_boundaries(cfg, root)
    return cfg


def _validate_protocol(cfg: dict[str, Any]) -> None:
    exp = cfg.get("experiment", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    name = str(exp.get("name", ""))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise InputValidationError(
            "experiment.name must contain only letters, digits, underscores, and hyphens."
        )
    if exp.get("overwrite", False):
        raise InputValidationError("experiment.overwrite must remain false for this project.")
    if data.get("use_validation_set", False):
        raise InputValidationError("Validation-set use is forbidden in Stage 0.")
    if data.get("allow_pixel_gt", False):
        raise InputValidationError("Pixel GT is forbidden in Stage 0.")
    if data.get("allow_oracle", False):
        raise InputValidationError("Oracle candidates are forbidden in Stage 0.")
    if str(data.get("split", "")).lower() != "train":
        raise InputValidationError("Only the train split is allowed.")
    if not model.get("normalize_features", True):
        raise InputValidationError("L2 normalization must be enabled.")
    if int(model.get("feature_dim", 0)) != 512:
        raise InputValidationError("This registered probe expects 512-dimensional features.")
    classes = data.get("classes", [])
    names = [str(item["name"]) for item in classes]
    if len(names) != len(set(names)) or not names:
        raise InputValidationError("Configured class names must be non-empty and unique.")
    class_ids = [int(item["id"]) for item in classes]
    if len(class_ids) != len(set(class_ids)):
        raise InputValidationError("Configured class IDs must be unique.")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_path_boundaries(cfg: dict[str, Any], project_root: Path) -> None:
    output_value = cfg.get("paths", {}).get("output_root")
    if not output_value:
        raise InputValidationError("paths.output_root is required.")
    output_root = Path(output_value).resolve()
    allowed_root = (project_root / "outputs").resolve()
    if output_root == allowed_root or not _is_relative_to(output_root, allowed_root):
        raise InputValidationError(
            f"output_root must be a named subdirectory of {allowed_root}, got {output_root}."
        )
    for key, value in cfg.get("paths", {}).items():
        if key == "output_root" or value is None:
            continue
        source = Path(value).resolve()
        if source == output_root or _is_relative_to(source, output_root) or _is_relative_to(output_root, source):
            raise InputValidationError(
                f"Output/input path overlap is forbidden: output_root={output_root}, {key}={source}."
            )


def write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def create_run_dir(output_root: str | Path, now: datetime | None = None) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d")
    for index in range(1, 10_000):
        candidate = root / f"run_{stamp}_{index:03d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate a unique run directory under {root}")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_entry(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {"configured": False, "exists": False}
    path = Path(path_value)
    entry: dict[str, Any] = {"configured": True, "path": str(path), "exists": path.exists()}
    if not path.exists():
        return entry
    stat = path.stat()
    entry.update({
        "kind": "directory" if path.is_dir() else "file",
        "size_bytes": stat.st_size,
        "modified_time_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    })
    if path.is_file():
        entry["sha256"] = sha256_file(path)
    return entry


def build_input_manifest(cfg: dict[str, Any], include_hashes: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in cfg.get("paths", {}).items():
        if key == "output_root":
            continue
        if include_hashes:
            result[key] = manifest_entry(value)
            if value and Path(value).is_file():
                path = Path(value)
                if path.suffix.lower() == ".json":
                    try:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        payload = {}
                    companion_name = payload.get("data_file") if isinstance(payload, dict) else None
                    if companion_name:
                        companion = path.parent / str(companion_name)
                        result[key]["data_companion"] = manifest_entry(str(companion))
                elif path.suffix.lower() == ".npz":
                    companion = path.with_suffix(".json")
                    if companion.is_file():
                        result[key]["metadata_companion"] = manifest_entry(str(companion))
        else:
            path = Path(value) if value else None
            result[key] = {
                "configured": value is not None,
                "path": str(path) if path else None,
                "exists": bool(path and path.exists()),
            }
    return result


def environment_text() -> str:
    lines = [
        f"python={sys.version.replace(chr(10), ' ')}",
        f"executable={sys.executable}",
        f"platform={platform.platform()}",
    ]
    modules = ["torch", "torchvision", "numpy", "pandas", "yaml", "matplotlib", "seaborn", "sklearn", "pytest", "open_clip"]
    for name in modules:
        try:
            module = __import__(name)
            lines.append(f"{name}={getattr(module, '__version__', 'unknown')}")
        except Exception as exc:
            lines.append(f"{name}=MISSING ({type(exc).__name__})")
    try:
        import torch
        lines.extend([
            f"torch_cuda={torch.version.cuda}",
            f"cuda_available={torch.cuda.is_available()}",
            f"cuda_device_count={torch.cuda.device_count()}",
            f"cuda_device_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
        ])
    except Exception:
        pass
    return "\n".join(lines) + "\n"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _decode_string_list(value: Any, field: str) -> list[str]:
    array = np.asarray(value).reshape(-1)
    result = []
    for item in array:
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        result.append(str(item))
    if not result:
        raise InputValidationError(f"{field} may not be empty.")
    return result


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if suffix == ".npy":
        return {"features": np.load(path, allow_pickle=False)}
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise InputValidationError("JSON feature input must be an object.")
        return value
    if suffix in {".pt", ".pth"}:
        import torch
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, dict):
            raise InputValidationError("Torch feature input must be a mapping.")
        return {key: item.detach().cpu().numpy() if hasattr(item, "detach") else item for key, item in value.items()}
    raise InputValidationError(f"Unsupported input format: {path.suffix}")


def _native_first_paper_mapping(
    path: Path,
    mapping: dict[str, Any],
    cfg: dict[str, Any],
    kind: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Adapt the first paper's JSON+NPZ prototype format without writing it."""
    metadata: dict[str, Any] | None = None
    if path.suffix.lower() == ".json" and "data_file" in mapping:
        metadata = mapping
        data_path = path.parent / str(metadata["data_file"])
        if not data_path.is_file():
            raise InputValidationError(f"Native calibration data file is missing: {data_path}")
        mapping = _load_mapping(data_path)
    elif path.suffix.lower() == ".npz":
        sidecar = path.with_suffix(".json")
        if sidecar.is_file():
            candidate = _load_mapping(sidecar)
            if candidate.get("data_file") == path.name:
                metadata = candidate

    if "prototypes" not in mapping:
        return mapping, None
    if metadata is None:
        raise InputValidationError(
            "First-paper prototype NPZ requires its JSON metadata sidecar for provenance validation."
        )
    class_id_key = "class_ids" if kind == "single" else "prototype_class_ids"
    if class_id_key not in mapping:
        raise InputValidationError(f"Native {kind} calibration lacks {class_id_key!r}.")
    id_to_name = {int(item["id"]): str(item["name"]) for item in cfg["data"]["classes"]}
    ids = np.asarray(mapping[class_id_key]).reshape(-1)
    try:
        names = [id_to_name[int(value)] for value in ids]
    except KeyError as exc:
        raise InputValidationError(f"Native calibration contains unmapped class id {exc.args[0]}.") from exc

    protocol = metadata.get("protocol", {})
    inputs = metadata.get("inputs", {})
    joined_inputs = " ".join(str(value).lower() for value in inputs.values())
    method = str(protocol.get("method", ""))
    if protocol.get("pixel_gt_used") is not False:
        raise InputValidationError("Native calibration must explicitly declare pixel_gt_used=false.")
    if protocol.get("love_da_val_used", False) is not False:
        raise InputValidationError("Native calibration indicates LoveDA Val use.")
    if "oracle" in method.lower() or "oracle" in path.name.lower():
        raise InputValidationError("Oracle prototype calibrations are forbidden.")
    if "loveda" not in joined_inputs or "train" not in joined_inputs:
        raise InputValidationError(
            "Native calibration metadata does not establish LoveDA Train input provenance."
        )
    if int(metadata.get("feature_dimension", -1)) != int(cfg["model"]["feature_dim"]):
        raise InputValidationError("Native calibration feature dimension does not match the probe.")
    metadata_classes = metadata.get("classes", {})
    if set(metadata_classes) != set(id_to_name.values()):
        raise InputValidationError("Native calibration class mapping differs from the registered classes.")

    adapted: dict[str, Any] = dict(mapping)
    adapted["features"] = mapping["prototypes"]
    adapted["class_names"] = np.asarray(names)
    if kind == "single":
        adapted["sample_counts"] = np.asarray(
            [
                metadata_classes[name].get(
                    "retained_seeds", metadata_classes[name].get("robust_retained_seeds", 0)
                )
                for name in names
            ],
            dtype=np.int64,
        )
    else:
        cluster_offsets: dict[str, int] = {name: 0 for name in metadata_classes}
        cluster_sizes: list[int] = []
        prototype_ids: list[str] = []
        for name in names:
            index = cluster_offsets[name]
            sizes = metadata_classes[name].get("cluster_sizes", [])
            if index >= len(sizes):
                raise InputValidationError(f"Native multi calibration lacks cluster size for {name}[{index}].")
            cluster_sizes.append(int(sizes[index]))
            prototype_ids.append(f"{name}_{index}")
            cluster_offsets[name] += 1
        adapted["cluster_sizes"] = np.asarray(cluster_sizes, dtype=np.int64)
        adapted["prototype_ids"] = np.asarray(prototype_ids)
    provenance = {
        "dataset": cfg["data"]["dataset_name"],
        "split": "train",
        "uses_pixel_gt": False,
        "uses_oracle": False,
        "construction": "; ".join(
            value for value in (method, str(protocol.get("seed_rule", ""))) if value
        ),
        "native_first_paper_metadata": metadata,
    }
    adapted["provenance"] = provenance
    return adapted, metadata


def _read_provenance(path: Path, mapping: dict[str, Any]) -> dict[str, Any] | None:
    raw = mapping.get("provenance")
    if raw is None:
        raw = mapping.get("provenance_json")
    if raw is not None:
        if isinstance(raw, np.ndarray) and raw.size == 1:
            raw = raw.reshape(-1)[0]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            raw = json.loads(raw)
        if isinstance(raw, dict):
            return raw
    sidecar = Path(str(path) + ".provenance.json")
    if sidecar.exists():
        with sidecar.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            return value
    return None


def _validate_provenance(provenance: dict[str, Any] | None, cfg: dict[str, Any], synthetic: bool) -> dict[str, Any]:
    if synthetic:
        return {"synthetic": True, "scientific_evidence": False}
    if provenance is None:
        if cfg["data"].get("require_provenance", True):
            raise InputValidationError("Formal inputs require provenance metadata.")
        return {"unverified": True}
    expected_dataset = str(cfg["data"]["dataset_name"]).lower()
    if str(provenance.get("dataset", "")).lower() != expected_dataset:
        raise InputValidationError("Provenance dataset does not match configured dataset.")
    if str(provenance.get("split", "")).lower() != "train":
        raise InputValidationError("Input provenance must declare split=train.")
    if provenance.get("uses_pixel_gt") is not False:
        raise InputValidationError("Input must explicitly declare uses_pixel_gt=false.")
    if provenance.get("uses_oracle") is not False:
        raise InputValidationError("Input must explicitly declare uses_oracle=false.")
    if not provenance.get("construction"):
        raise InputValidationError("Input provenance must describe its construction rule.")
    return provenance


def load_feature_bundle(path_value: str, cfg: dict[str, Any], kind: str, synthetic: bool = False) -> FeatureBundle:
    path = Path(path_value)
    if not path.is_file():
        raise InputValidationError(f"Missing {kind} input: {path}")
    mapping = _load_mapping(path)
    native_metadata = None
    if kind in {"single", "multi"}:
        mapping, native_metadata = _native_first_paper_mapping(path, mapping, cfg, kind)
    if "features" not in mapping:
        raise InputValidationError(f"{kind} input lacks 'features'.")
    features = np.asarray(mapping["features"], dtype=np.float32)
    expected_dim = int(cfg["model"]["feature_dim"])
    if features.ndim != 2 or features.shape[1] != expected_dim:
        raise InputValidationError(f"{kind} features must have shape [N,{expected_dim}], got {features.shape}.")
    if not np.isfinite(features).all():
        raise InputValidationError(f"{kind} features contain NaN or infinity.")
    if "class_names" not in mapping and kind != "region":
        raise InputValidationError(f"{kind} input lacks explicit class_names.")
    names = _decode_string_list(mapping.get("class_names", []), "class_names") if "class_names" in mapping else ["unknown"] * len(features)
    if len(names) != len(features):
        raise InputValidationError(f"{kind} class_names length does not match features.")
    configured = {str(item["name"]) for item in cfg["data"]["classes"]}
    unknown = sorted(set(names) - configured - {"unknown"})
    if unknown:
        raise InputValidationError(f"{kind} contains unmapped classes: {unknown}")
    provenance = _validate_provenance(_read_provenance(path, mapping), cfg, synthetic)
    metadata = {
        "path": str(path.resolve()),
        "provenance": provenance,
        "shape": list(features.shape),
        "input_norm_mean": float(np.linalg.norm(features, axis=1).mean()),
        "input_norm_std": float(np.linalg.norm(features, axis=1).std()),
        "native_first_paper_format": native_metadata is not None,
    }
    def optional_array(key: str) -> np.ndarray | None:
        if key not in mapping:
            return None
        value = np.asarray(mapping[key]).reshape(-1)
        if len(value) != len(features):
            raise InputValidationError(f"{kind} field {key!r} length does not match features.")
        return value
    prototype_ids = _decode_string_list(mapping["prototype_ids"], "prototype_ids") if "prototype_ids" in mapping else None
    cam_labels = _decode_string_list(mapping["cam_labels"], "cam_labels") if "cam_labels" in mapping else None
    sam3_labels = _decode_string_list(mapping["sam3_source_labels"], "sam3_source_labels") if "sam3_source_labels" in mapping else None
    for field, value in (("prototype_ids", prototype_ids), ("cam_labels", cam_labels), ("sam3_source_labels", sam3_labels)):
        if value is not None and len(value) != len(features):
            raise InputValidationError(f"{kind} field {field!r} length does not match features.")
    return FeatureBundle(
        features=features,
        class_names=names,
        metadata=metadata,
        sample_counts=optional_array("sample_counts"),
        prototype_ids=prototype_ids,
        cluster_sizes=optional_array("cluster_sizes"),
        cam_labels=cam_labels,
        sam3_source_labels=sam3_labels,
    )


def load_region_bundle(feature_path: str, label_path: str | None, cfg: dict[str, Any]) -> FeatureBundle:
    if Path(feature_path).is_dir():
        if label_path:
            raise InputValidationError(
                "Native directory inputs derive CAM/SAM3 labels by keyed read-only joining; "
                "a separate weak_region_label_file is forbidden."
            )
        from .native_region import load_native_region_directory

        return load_native_region_directory(cfg)
    raise InputValidationError(
        "Generic formal region bundles are disabled because they cannot prove keyed, "
        "non-circular label derivation. Use the native region/candidate/CAM directory adapter."
    )


def inspect_configured_inputs(cfg: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {"ready": True, "items": {}}
    required = ["remoteclip_checkpoint", "single_prototype_file"]
    optional = ["multi_prototype_file", "region_feature_cache", "weak_region_label_file"]
    for key in required + optional:
        value = cfg["paths"].get(key)
        candidate = Path(value) if value else None
        exists = bool(candidate and (candidate.is_file() or (key == "region_feature_cache" and candidate.is_dir())))
        status = "ready" if exists else ("missing_required" if key in required else "not_configured_optional")
        checks["items"][key] = {"path": value, "status": status}
        if key in required and not exists:
            checks["ready"] = False
    text_cache = cfg["paths"].get("text_feature_cache")
    if text_cache and Path(text_cache).is_file() and Path(text_cache).with_suffix(".json").is_file():
        checks["items"]["text_encoder"] = {"status": "ready_cached", "path": text_cache}
    else:
        try:
            import open_clip  # noqa: F401
            checks["items"]["text_encoder"] = {"status": "ready_local_open_clip"}
        except Exception:
            checks["items"]["text_encoder"] = {"status": "missing_required", "path": text_cache}
            checks["ready"] = False
    return checks

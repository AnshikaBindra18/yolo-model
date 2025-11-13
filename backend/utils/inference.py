import base64
import io
from functools import lru_cache
from typing import Tuple, Dict, Any
from pathlib import Path
import os

import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel, SegmentationModel, ClassificationModel
from ultralytics.nn.modules.conv import Conv as UltralyticsConv
try:
    from ultralytics.nn.modules.block import C3k2 as UltralyticsC3k2
except Exception:
    UltralyticsC3k2 = None
# PyTorch 2.6 switched torch.load default to weights_only=True. When Ultralytics checkpoints
# include model class references, we must allowlist them for safe unpickling.
from torch.nn.modules.container import Sequential, ModuleList, ModuleDict
_safe_globals_items = [
    # Ultralytics task models
    DetectionModel, SegmentationModel, ClassificationModel,
    # Specific Ultralytics conv module referenced by some checkpoints
    UltralyticsConv,
    # Specific Ultralytics block classes referenced by checkpoints (may be None)
    UltralyticsC3k2,
    # Common PyTorch container modules referenced in checkpoints
    Sequential, ModuleList, ModuleDict,
]
# Filter out any None entries before registering
_safe_globals_items = [x for x in _safe_globals_items if x is not None]
if _safe_globals_items:
    add_safe_globals(_safe_globals_items)

# Broadly allowlist most torch.nn module classes to reduce repeated failures
# when loading trusted local checkpoints under PyTorch 2.6 weights-only mode.
def _allowlist_torch_nn_modules():
    try:
        import inspect
        import types
        nn_modules = []
        def _collect(module):
            for name in dir(module):
                try:
                    obj = getattr(module, name)
                except Exception:
                    continue
                if inspect.isclass(obj):
                    try:
                        if issubclass(obj, torch.nn.Module):
                            nn_modules.append(obj)
                    except Exception:
                        pass
                elif isinstance(obj, types.ModuleType) and obj.__name__.startswith('torch.nn'):
                    _collect(obj)
        _collect(torch.nn)
        if nn_modules:
            add_safe_globals(list(set(nn_modules)))
    except Exception:
        # Best-effort; continue even if inspection fails
        pass

_allowlist_torch_nn_modules()


def _allowlist_ultralytics_modules():
    """Best-effort: collect and allowlist classes defined under
    ultralytics.nn.modules so torch's WeightsUnpickler accepts them
    when loading weights_only checkpoints. This avoids repeatedly
    adding single classes when Ultralytics checkpoints reference
    many small module classes (e.g., C3k2).

    We only register classes that subclass torch.nn.Module and skip
    anything that fails to import/inspect.
    """
    try:
        import inspect
        import ultralytics
        from ultralytics import nn as u_nn

        ultralytics_classes = []

        def _collect(module):
            for name in dir(module):
                try:
                    obj = getattr(module, name)
                except Exception:
                    continue
                # If it's a class and a subclass of torch.nn.Module, collect it
                if inspect.isclass(obj):
                    try:
                        if issubclass(obj, torch.nn.Module):
                            ultralytics_classes.append(obj)
                    except Exception:
                        pass
                # If it's a submodule, recurse once
                elif inspect.ismodule(obj) and obj.__name__.startswith('ultralytics.nn'):
                    _collect(obj)

        # Start from ultralytics.nn.modules if present, else from ultralytics.nn
        root = getattr(u_nn, 'modules', None) or u_nn
        _collect(root)

        if ultralytics_classes:
            # de-duplicate and register
            add_safe_globals(list(set(ultralytics_classes)))
    except Exception:
        # Best-effort: don't crash on allowlisting
        pass


_allowlist_ultralytics_modules()


def _allowlist_by_name(full_names):
    """Allowlist globals by their full module-qualified name even when
    the actual class object isn't importable as an attribute. This
    creates lightweight dummy types with the same __module__ and
    __qualname__ so PyTorch's weights-only unpickler recognizes them.

    Use with care: this skips checking that the real implementation
    exists and should only be used for trusted local checkpoints.
    """
    created = []
    for fn in full_names:
        try:
            module, name = fn.rsplit('.', 1)
        except Exception:
            continue
        try:
            # Make a lightweight placeholder type with matching identity
            T = type(name, (), {})
            T.__module__ = module
            T.__qualname__ = name
            created.append(T)
        except Exception:
            continue
    if created:
        try:
            add_safe_globals(created)
        except Exception:
            pass


# Handle known dynamic class names referenced in Ultralytics checkpoints
_allowlist_by_name([
    'ultralytics.nn.modules.block.C3k2',
    'ultralytics.nn.modules.block.C3k',
    'ultralytics.nn.modules.block.C2PSA',
    'ultralytics.nn.modules.block.C3TR',
    'ultralytics.nn.modules.block.C2f',
])


VEGETATION_MODEL_PATH = str((__file__).replace('utils\\inference.py', 'models\\vegetation_model.pt'))

# For isolation, prefer loading only the SOIL model. Use an env override first for explicit path,
# else try yolo11m.pt in the models folder, else fall back to soil_model.pt.
_models_dir = Path(__file__).resolve().parent.parent / 'models'
_env_soil = os.getenv('SOIL_MODEL_PATH')
_soil_candidates = []
if _env_soil:
    p = Path(_env_soil)
    if p.exists():
        _soil_candidates.append(p)
_soil_candidates.extend([
    _models_dir / 'yolo11m.pt',
    _models_dir / 'soil_model.pt',
])
_soil_path = next((p for p in _soil_candidates if Path(p).exists()), _models_dir / 'soil_model.pt')
SOIL_MODEL_PATH = str(_soil_path)


@lru_cache(maxsize=1)
def get_vegetation_model():
    print("[veg] Loading model from:", VEGETATION_MODEL_PATH)
    print("[veg] Model path exists?", Path(VEGETATION_MODEL_PATH).exists())
    try:
        model = YOLO(VEGETATION_MODEL_PATH)
        print("[veg] Model loaded successfully")
        
        print("[veg] Model configuration:")
        if hasattr(model, "overrides"):
            print("  - Task:", model.overrides.get("task", "unknown"))
            print("  - Mode:", model.overrides.get("mode", "unknown"))
        print("  - Model type:", type(model).__name__)
        if hasattr(model, "model"):
            print("  - Architecture:", type(model.model).__name__)
        if hasattr(model, "names"):
            print("  - Class names:", model.names)
        
        # Optionally run a lightweight test inference to surface runtime issues.
        # Wrap in try/except because some Ultralytics internal versions may
        # attempt to call legacy methods (e.g. .detect) on architecture
        # objects and raise AttributeError even though the model object
        # itself is usable via .predict() in request-time execution.
        try:
            import numpy as np
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            print("[veg] Running test inference...")
            test_results = model.predict(test_img, verbose=False)
            if test_results:
                r0 = test_results[0]
                print("  - Returns boxes?", hasattr(r0, "boxes"))
                print("  - Returns masks?", hasattr(r0, "masks"))
                print("  - Returns probs?", hasattr(r0, "probs"))
        except Exception as te:
            # Log but don't fail model loading: the model object was created
            # successfully and can still be used at request time with .predict().
            print(f"[veg] Test inference raised an exception (non-fatal): {te}")
            
        return model
    except Exception as e:
        print("[veg] Error loading model:", e)
        raise


@lru_cache(maxsize=1)
def get_soil_model():
    # Diagnostics to help debug loading issues under PyTorch 2.6.
    try:
        print(f"[soil] Loading model from: {SOIL_MODEL_PATH}")
        print(f"[soil] torch version={torch.__version__}")
        try:
            import ultralytics
            print(f"[soil] ultralytics version={ultralytics.__version__}")
        except Exception:
            pass
        model = YOLO(SOIL_MODEL_PATH)
        print("[soil] Model loaded OK")
        return model
    except Exception as e:
        print(f"[soil] Model load FAILED: {e}")

        # If the checkpoint is trusted (local development), allow an
        # explicit unsafe fallback to load the checkpoint with
        # weights_only=False. This can execute arbitrary code from the
        # checkpoint, so only do it when the environment variable
        # ALLOW_UNSAFE_CKPT_LOAD is set to '1'. If not set, re-raise
        # the original exception and instruct the user on next steps.
        allow_unsafe = os.getenv('ALLOW_UNSAFE_CKPT_LOAD', '0') == '1'
        if allow_unsafe:
            print('[soil] ALLOW_UNSAFE_CKPT_LOAD=1: attempting unsafe load (weights_only=False)')
            _orig_torch_load = torch.load

            def _torch_load_force_safe(*args, **kwargs):
                # Force weights_only=False unless explicitly provided.
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return _orig_torch_load(*args, **kwargs)

            try:
                torch.load = _torch_load_force_safe
                model = YOLO(SOIL_MODEL_PATH)
                print('[soil] Model loaded OK (unsafe mode)')
                return model
            except Exception as e2:
                print(f"[soil] Unsafe load also failed: {e2}")
                raise
            finally:
                # restore original
                torch.load = _orig_torch_load

        # Not allowed to do unsafe load: surface guidance to the caller
        raise


def _read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    print("[debug] Original image mode:", image.mode)
    print("[debug] Original image size:", image.size)
    image = image.convert("RGB")
    np_img = np.array(image)
    print("[debug] Numpy array shape:", np_img.shape)
    print("[debug] Numpy array dtype:", np_img.dtype)
    print("[debug] Value range:", np_img.min(), "to", np_img.max())
    return np_img  # RGB uint8


def _to_base64_image(np_image: np.ndarray, format: str = "PNG") -> str:
    image_pil = Image.fromarray(np_image)
    buffer = io.BytesIO()
    image_pil.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def analyze_vegetation(image_bytes: bytes) -> Dict[str, Any]:
    model = get_vegetation_model()
    img = _read_image_from_bytes(image_bytes)

    print("[veg] Running prediction on image shape:", img.shape)
    
    try:
        results = model.predict(
            source=img,
            conf=0.1,  # Even lower confidence threshold for testing
            verbose=True  # Enable verbose mode to see what's happening
        )
    except Exception as e:
        # Surface a clear error for the endpoint while keeping the
        # original traceback logged by the server.
        print(f"[veg] Prediction failed: {e}")
        raise RuntimeError(f"Vegetation prediction failed: {e}")
    
    if not results:
        print("[veg] No results from predict()")
        return {"vegetation": False, "mask_image_base64": None}
        
    print("[veg] Got prediction results:")
    r0 = results[0]
    print("  - Has boxes?", hasattr(r0, "boxes"))
    if hasattr(r0, "boxes") and r0.boxes is not None:
        print("  - Number of boxes:", len(r0.boxes))
        if len(r0.boxes) > 0:
            print("  - First box confidence:", float(r0.boxes[0].conf[0]))
            print("  - First box class:", int(r0.boxes[0].cls[0].item()))
    print("  - Has masks?", hasattr(r0, "masks") and r0.masks is not None)
    if hasattr(r0, "masks") and r0.masks is not None:
        print("  - Number of masks:", len(r0.masks))

    r0 = results[0]
    print("[veg] Got results. Has masks attr?", hasattr(r0, "masks"))
    
    if hasattr(r0, "boxes") and r0.boxes is not None:
        print("[veg] Number of detections:", len(r0.boxes))
        print("[veg] First box conf:", float(r0.boxes[0].conf[0]) if len(r0.boxes) > 0 else "N/A")
    
    masks_attr = getattr(r0, "masks", None)
    print("[veg] Masks present?", masks_attr is not None)
    print("[veg] Masks has data?", masks_attr.data is not None if masks_attr else False)
    print("[veg] Number of masks:", len(masks_attr.data) if masks_attr and masks_attr.data is not None else 0)

    # Check for either masks or high-confidence boxes
    has_masks = (
        getattr(r0, "masks", None) is not None and 
        r0.masks is not None and 
        r0.masks.data is not None and 
        len(r0.masks.data) > 0
    )
    
    has_confident_boxes = (
        hasattr(r0, "boxes") and 
        r0.boxes is not None and 
        len(r0.boxes) > 0 and 
        any(float(box.conf[0]) > 0.3 for box in r0.boxes)  # At least one box with >30% confidence
    )
    
    # Consider vegetation present if we have either masks or confident boxes
    is_vegetation = bool(has_masks or has_confident_boxes)
    
    print("[veg] Detection result:", {
        "has_masks": bool(has_masks),
        "has_confident_boxes": bool(has_confident_boxes),
        "is_vegetation": bool(is_vegetation)
    })

    # Use Ultralytics built-in plot to draw detection visualization
    overlay_img = r0.plot()  # returns a BGR numpy array by default
    if overlay_img is None:
        overlay_img = img

    # Ensure RGB for PIL
    if overlay_img.shape[-1] == 3:
        overlay_img = overlay_img[:, :, ::-1]  # BGR -> RGB

    overlay_base64 = _to_base64_image(overlay_img)

    return {
        "vegetation": bool(is_vegetation),
        "mask_image_base64": overlay_base64,
        "debug_info": {
            "has_masks": bool(has_masks),
            "has_confident_boxes": bool(has_confident_boxes),
            "num_detections": int(len(r0.boxes)) if hasattr(r0, "boxes") and r0.boxes is not None else 0,
            "num_masks": int(len(r0.masks.data)) if has_masks else 0
        }
    }


def analyze_soil(image_bytes: bytes) -> Dict[str, Any]:
    model = get_soil_model()
    img = _read_image_from_bytes(image_bytes)

    try:
        results = model.predict(source=img, verbose=False)
    except Exception as e:
        print(f"[soil] Prediction failed: {e}")
        raise RuntimeError(f"Soil prediction failed: {e}")

    if not results:
        return {"soil_type": None, "confidence": 0.0}

    r0 = results[0]

    # Classification probabilities
    probs = getattr(r0, "probs", None)
    if probs is None:
        # In case the model returns detections with class, try alternate extraction
        names = r0.names if hasattr(r0, "names") else {}
        if hasattr(r0, "boxes") and r0.boxes is not None and len(r0.boxes) > 0:
            cls_id = int(r0.boxes.cls[0].item())
            conf = float(r0.boxes.conf[0].item())
            label = names.get(cls_id, str(cls_id))
            return {"soil_type": label, "confidence": conf}
        return {"soil_type": None, "confidence": 0.0}

    # Ensure native Python types
    top1_id = int(probs.top1)
    top1_conf = float(probs.top1conf)
    names = r0.names if hasattr(r0, "names") else {}
    label = names.get(top1_id, str(top1_id))

    # Optionally normalize label to expected set
    label_map = {
        "black soil": "Black soil",
        "black": "Black soil",
        "clay soil": "Clay soil",
        "clay": "Clay soil",
        "red soil": "Red soil",
        "red": "Red soil",
        "alluvial soil": "Alluvial soil",
        "alluvial": "Alluvial soil",
    }
    norm_label = label_map.get(label.lower(), label)

    return {
        "soil_type": str(norm_label),
        "confidence": float(top1_conf),
    }


def analyze_soil_test(image_bytes: bytes) -> Dict[str, Any]:
    """
    Temporary test utility to isolate soil model loading/inference.
    Returns a list of class names with confidence scores, without any vegetation logic.

    Rationale: isolate model-loading issues by loading ONLY the soil model first,
    verify end-to-end inference, then re-introduce vegetation model next.
    """
    model = get_soil_model()
    img = _read_image_from_bytes(image_bytes)

    try:
        results = model.predict(source=img, verbose=False)
    except Exception as e:
        print(f"[soil-test] Prediction failed: {e}")
        raise RuntimeError(f"Soil test prediction failed: {e}")

    if not results:
        return {"classes": [], "top": None}

    r0 = results[0]
    names = r0.names if hasattr(r0, "names") else {}

    out = []
    probs = getattr(r0, "probs", None)
    if probs is not None:
        # For classification models, use probs to extract class distribution
        # Get all confidences and sort desc
        confs = probs.data.tolist() if hasattr(probs, 'data') else []
        for cls_id, conf in enumerate(confs):
            if conf is None:
                continue
            out.append({
                "name": names.get(cls_id, str(cls_id)),
                "confidence": float(conf),
            })
        out.sort(key=lambda x: x["confidence"], reverse=True)
    else:
        # If detections with boxes are returned, aggregate by class (max conf)
        if hasattr(r0, "boxes") and r0.boxes is not None and len(r0.boxes) > 0:
            agg: Dict[int, float] = {}
            for i in range(len(r0.boxes)):
                cls_id = int(r0.boxes.cls[i].item())
                conf = float(r0.boxes.conf[i].item())
                agg[cls_id] = max(agg.get(cls_id, 0.0), conf)
            for cls_id, conf in agg.items():
                out.append({
                    "name": names.get(cls_id, str(cls_id)),
                    "confidence": float(conf),
                })
            out.sort(key=lambda x: x["confidence"], reverse=True)

    top = out[0] if out else None
    return {"classes": out, "top": top}


def analyze_vegetation_test(image_bytes: bytes) -> Dict[str, Any]:
    """
    Detailed test utility for vegetation model inference, similar to analyze_soil_test.
    Returns a list of detection results with confidence scores and box coordinates.

    Returns:
        Dict with fields:
        - detections: List of dict with fields:
            - confidence: float, detection confidence
            - class_name: str, predicted class
            - has_mask: bool, whether segmentation mask exists
            - bbox: dict with x1,y1,x2,y2 (normalized 0-1)
    """
    model = get_vegetation_model()
    img = _read_image_from_bytes(image_bytes)

    try:
        results = model.predict(source=img, verbose=False)
    except Exception as e:
        print(f"[veg-test] Prediction failed: {e}")
        raise RuntimeError(f"Vegetation test prediction failed: {e}")

    if not results:
        return {"detections": [], "mask_image_base64": None}

    r0 = results[0]
    names = r0.names if hasattr(r0, "names") else {}

    detections = []
    # Get mask overlay if available
    mask_overlay = None
    if hasattr(r0, "plot"):
        try:
            overlay_img = r0.plot()
            if overlay_img is not None:
                if overlay_img.shape[-1] == 3:
                    overlay_img = overlay_img[:, :, ::-1]  # BGR -> RGB
                mask_overlay = _to_base64_image(overlay_img)
        except Exception:
            pass

    # Extract boxes and masks if present
    if hasattr(r0, "boxes") and r0.boxes is not None and len(r0.boxes) > 0:
        for i in range(len(r0.boxes)):
            box = r0.boxes[i]
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            # Get normalized bbox coordinates (0-1 range)
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            img_h, img_w = img.shape[:2]
            x1, x2 = x1/img_w, x2/img_w
            y1, y2 = y1/img_h, y2/img_h
            
            # Check if this detection has an associated mask
            has_mask = (
                hasattr(r0, "masks") and
                r0.masks is not None and
                r0.masks.data is not None and
                i < len(r0.masks.data)
            )
            
            detections.append({
                "confidence": conf,
                "class_name": names.get(cls_id, str(cls_id)),
                "has_mask": has_mask,
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            })

    # Sort by confidence
    detections.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "detections": detections,
        "mask_image_base64": mask_overlay,
    }
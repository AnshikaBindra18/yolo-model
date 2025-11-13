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

    results = model.predict(source=img, verbose=False)
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
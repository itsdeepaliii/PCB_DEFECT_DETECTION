import cv2
from backend.image_processing import generate_defect_mask
from backend.classifier import classify_roi


def run_full_pipeline(test_img_path, template_img_path, model, class_names):

    test_img = cv2.imread(test_img_path)
    template_img = cv2.imread(template_img_path)

    if test_img is None or template_img is None:
        return None, []

    # Resize template to match test
    template_img = cv2.resize(
        template_img,
        (test_img.shape[1], test_img.shape[0])
    )

    # Generate defect mask
    mask = generate_defect_mask(test_img, template_img)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    annotated = test_img.copy()
    predictions = []

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area < 40:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 8 or h < 8:
            continue

        pad = 20
        y1 = max(0, y-pad)
        y2 = min(test_img.shape[0], y+h+pad)
        x1 = max(0, x-pad)
        x2 = min(test_img.shape[1], x+w+pad)

        roi = test_img[y1:y2, x1:x2]

        label, conf = classify_roi(model, roi, class_names)

        if conf < 0.85:
            continue

        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(annotated,
                    f"{label} {conf*100:.1f}%",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2)

        predictions.append({
            "label": label,
            "confidence": round(conf*100, 2),
            "bbox": [x, y, w, h]
        })

    return annotated, predictions
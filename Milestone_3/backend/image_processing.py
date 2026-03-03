import cv2
import numpy as np
import os

def find_best_template(test_img, template_dir):
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    test_small = cv2.resize(gray_test, (256, 256))

    best_score = -1
    best_template = None

    for name in os.listdir(template_dir):
        path = os.path.join(template_dir, name)
        temp = cv2.imread(path, 0)
        if temp is None:
            continue

        temp_small = cv2.resize(temp, (256, 256))
        result = cv2.matchTemplate(test_small, temp_small, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_template = path

    return best_template


def generate_defect_mask(test_img, template_img):
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    gray_temp = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_test = clahe.apply(gray_test)
    gray_temp = clahe.apply(gray_temp)

    diff = cv2.absdiff(gray_temp, gray_test)

    mask = cv2.adaptiveThreshold(
        diff, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    mask = cv2.medianBlur(mask, 3)

    _, certainty = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    final_mask = cv2.bitwise_and(mask, certainty)

    return final_mask
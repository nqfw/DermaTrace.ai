import cv2

import numpy as np

def create_blue_mask(image):
    """
    Detect blue ruler and grid lines using HSV color detection.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean small noise in mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.medianBlur(mask, 5)
    return mask

def apply_dullrazor(img):
    """
    Applies the DullRazor algorithm to remove hair and other dark artifacts 
    from the image, using morphological operations and inpainting.
    Also removes blue rulers based on the user's provided logic.
    """
    # 1. Hair Removal (DullRazor)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # INCREASED THRESHOLD: 10 was too aggressive and captured red lesion patches and skin borders. 
    # Raising to 50 ensures it only captures very distinct, high-contrast dark hairs.
    _, hair_mask = cv2.threshold(blackhat, 50, 255, cv2.THRESH_BINARY)
    
    # 2. Ruler Removal (User's Blue Mask)
    ruler_mask = create_blue_mask(img)
    
    # Combine masks to save computational time on inpainting
    combined_mask = cv2.bitwise_or(hair_mask, ruler_mask)
    
    # 3. Inpaint the original image using the combined mask
    img_clean = cv2.inpaint(img, combined_mask, 3, cv2.INPAINT_TELEA)
    
    return img_clean

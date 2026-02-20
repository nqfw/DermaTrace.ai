import cv2

def apply_dullrazor(img):
    """
    Applies the DullRazor algorithm to remove hair and other dark artifacts 
    from the image, using morphological operations and inpainting.
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Blackhat Transform
    # Finds dark, thin structures (like hair) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 3. Create Mask of hairs
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # 4. Inpaint the original image using the hair mask
    img_clean = cv2.inpaint(img, hair_mask, 1, cv2.INPAINT_TELEA)
    
    return img_clean

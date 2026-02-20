import cv2
import sys
import numpy as np

# Import our processing modules
from skin import process_image
from dullrazor import apply_dullrazor

def audit_image(image_path):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
        
    # Standardize size for display
    img = cv2.resize(img, (400, 300))

    # 2. Apply DullRazor
    img_dullrazor = apply_dullrazor(img)

    # 3. Apply Skin Detection (on the DullRazor output)
    is_skin, pct, mask = process_image(img_dullrazor)
    
    # Convert mask to 3 channels so we can stack it horizontally with the colored images
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # 4. Create Side-by-Side Visualization
    # [Original | DullRazor | Skin Mask]
    combined = np.hstack((img, img_dullrazor, mask_bgr))
    
    # Add text overlay with results
    text = f"Skin Coverage: {pct:.1f}% | IS SKIN: {is_skin}"
    cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 5. Show Image
    cv2.imshow("Pipeline Audit (Original | DullRazor | Skin Mask)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audit_pipeline.py <path_to_image>")
        sys.exit(1)
        
    audit_image(sys.argv[1])

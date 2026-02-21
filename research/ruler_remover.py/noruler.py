import cv2
import numpy as np
import os


def create_blue_mask(image):
    """
    Detect blue ruler and grid lines using HSV color detection.
    """
    # Convert image from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Blue color range (for ruler & grid)
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([140, 255, 255])

    # Create mask of blue areas
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean small noise in mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.medianBlur(mask, 5)

    return mask


def remove_blue_ruler_and_grid(image):
    """
    Remove blue ruler and grid using inpainting.
    """
    mask = create_blue_mask(image)

    # Inpaint removes detected blue regions intelligently
    cleaned_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    return cleaned_image


def process_images(input_folder, output_folder):
    """
    Process all images from raw folder and save cleaned images.
    """
    if not os.path.exists(input_folder):
        print("ERROR: data/raw folder not found!")
        return

    os.makedirs(output_folder, exist_ok=True)

    total = 0

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            image = cv2.imread(input_path)

            if image is None:
                print(f"Skipping unreadable file: {filename}")
                continue

            cleaned = remove_blue_ruler_and_grid(image)
            cv2.imwrite(output_path, cleaned)

            total += 1
            print(f"Processed: {filename}")

    print("\n=========================")
    print(f"Total Images Processed: {total}")
    print("Blue Ruler & Grid Removed Successfully!")
    print("=========================")


if __name__ == "__main__":
    # Based on your project structure
    input_folder = "../data/raw/"
    output_folder = "../data/processed/"

    print("Starting Blue Ruler & Grid Removal...")
    process_images(input_folder, output_folder)
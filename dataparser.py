from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import pytesseract
import os
import easyocr
import numpy as np

def apply_ocr(comic_image):
    custom_config = r'--oem 3 --psm 6'  # Use LSTM OCR Engine and assume a single uniform block of text
    ocr_data = pytesseract.image_to_data(comic_image, output_type=pytesseract.Output.DICT, config=custom_config)

    # Remove detected text by drawing over it
    draw = ImageDraw.Draw(comic_image)
    for i in range(len(ocr_data['text'])):
        if ocr_data['text'][i].strip():  # If text is detected
            x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i],
                            ocr_data['width'][i], ocr_data['height'][i])
            text = ocr_data['text'][i]

            if text.strip().isalpha() and len(text) > 2:
                # print(text)
                draw.rectangle([x, y, x + w, y + h], fill="white")
 
def apply_easy_ocr(comic_image):
    # Convert the Pillow image to a NumPy array for EasyOCR
    comic_image_np = np.array(comic_image)
    
    reader = easyocr.Reader(['en'], gpu=True)  # Initialize EasyOCR reader
    ocr_results = reader.readtext(comic_image_np)
    # print(ocr_results)
    # Convert the NumPy array back to a Pillow image for drawing
    comic_image = Image.fromarray(comic_image_np)

    # Remove detected text by drawing over it
    draw = ImageDraw.Draw(comic_image)
    for (bbox, _, confidence) in ocr_results:  # Replace 'text' with '_' since it's unused
        # if confidence > 0.1:  # Filter out low-confidence detections
            (top_left, top_right, bottom_right, bottom_left) = bbox
            x_min = int(min(top_left[0], bottom_left[0]))
            y_min = int(min(top_left[1], top_right[1]))
            x_max = int(max(top_right[0], bottom_right[0]))
            y_max = int(max(bottom_left[1], bottom_right[1]))

            # Draw a white rectangle over the detected text
            draw.rectangle([x_min, y_min, x_max, y_max], fill="white")
    return comic_image
def preprocess_image(image):
    """
    Preprocess the image to improve OCR accuracy.
    Converts to grayscale, enhances contrast, and applies thresholding.
    """
    # Convert to grayscale
    grayscale = image.convert("L")
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(grayscale)
    enhanced = enhancer.enhance(1.35)
    # Apply thresholding
    thresholded = enhanced.point(lambda p: p > 150 and 255)
    return thresholded

def extract_comic_panels(image_path, output_dir, panel_width, panel_height, panel_count=0):
    """
    Extracts panels from a newspaper comic image, removes text using OCR, and saves them as separate images.

    Args:
        image_path (str): Path to the comic image.
        output_dir (str): Directory to save the extracted panels.
        panel_width (int): Width of each panel.
        panel_height (int): Height of each panel.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
        # Open the comic image
        comic_image = Image.open(image_path)
        comic_width, comic_height = comic_image.size
        
        # comic_image = preprocess_image(comic_image)
        
        # Apply OCR to remove text
        comic_image = apply_easy_ocr(comic_image)


        # Calculate the number of panels
        num_columns = comic_width // panel_width
        num_rows = comic_height // panel_height

        # Extract each panel
        for row in range(num_rows):
            for col in range(num_columns):
                left = col * panel_width
                upper = row * panel_height
                right = left + panel_width
                lower = upper + panel_height

                # Crop the panel
                panel = comic_image.crop((left, upper, right, lower))
                
                # Preprocess the panel for better OCR
                # preprocessed_panel = preprocess_image(panel)
                
                # Perform OCR to detect text
                

                # Save the panel
                panel_path = f"{output_dir}/panel_{panel_count}.png"
                print(panel_path)
                panel.save(panel_path)
                panel_count += 1

        print(f"Extracted {panel_count} panels successfully.")

    except Exception as e:
        print(f"Error: {e}")

# Example usage:
for i in range(1, 3000):
    if i%7 == 0:
        continue
    extract_comic_panels(f"mnt/calvin_hobbes_comics/{i}_ch.png", "OCR_PARSED_PANELS", 142, 183, (i-1)*4)
# extract_comic_panels("mnt/calvin_hobbes_comics/2_ch.png", "output_panels", 142, 188)
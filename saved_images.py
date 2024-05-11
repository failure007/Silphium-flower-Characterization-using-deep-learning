import cv2
import numpy as np
import os

# Function to perform semantic segmentation on flowers in an image
def segment_flowers(image_path):
    # Read the image
    img = cv2.imread(image_path)
   
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   
    # Define the lower and upper bounds for yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
   
    # Create a mask for yellow color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
   
    # Bitwise AND the original image with the mask
    segmented_img = cv2.bitwise_and(img, img, mask=mask)

    # Find contours in the segmented image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (main flower)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask for the main flower
        main_flower_mask = np.zeros_like(mask)
        cv2.drawContours(main_flower_mask, [main_contour], 0, 255, thickness=cv2.FILLED)
        
        # Bitwise AND the original image with the main flower mask
        main_flower_img = cv2.bitwise_and(img, img, mask=main_flower_mask)

        # Resize the original image to match the height of the main flower image
        img_resized = cv2.resize(img, (main_flower_img.shape[1], main_flower_img.shape[0]))

        # Display the original and main flower images side by side
        combined_img = np.hstack((img_resized, main_flower_img))

        # Resize combined image to fit the window exactly
        height, width, _ = combined_img.shape
        max_display_height = 800  # Set the maximum display height as needed
        scale_factor = min(1.0, max_display_height / height)
        combined_img_resized = cv2.resize(combined_img, (int(width * scale_factor), int(height * scale_factor)))

        cv2.imshow("Original vs Main Flower", combined_img_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Replace this path with the actual path to your images folder
images_folder_path = "C:/Users/DheerajKumarJallipal/Downloads/silphium/images"

# Perform semantic segmentation and display for each image in the images folder
for image_file in os.listdir(images_folder_path):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(images_folder_path, image_file)
        segment_flowers(image_path)

import os
import pandas as pd
import cv2

# Define number of samples to verify
NUM_SAMPLES = 730

# Load data from CSV file
data_path = "C:/Users/DheerajKumarJallipal/Desktop/label.csv"
data = pd.read_csv(data_path)

# Define image directory
image_dir = "C:/Users/DheerajKumarJallipal/Desktop/silphium/images"

# Sample random filenames for verification
sampled_filenames = data['filename'].sample(NUM_SAMPLES)

# Loop through samples
for filename in sampled_filenames:
  # Construct image path
  img_path = os.path.join(image_dir, f"{filename}.jpg")

  # Check if image exists
  if os.path.exists(img_path):
    # Load image
    img = cv2.imread(img_path)
    
    # Display image (optional)
    # cv2.imshow(filename, img)
    # cv2.waitKey(0)
    
    # Get count from dataframe
    expected_count = data[data['filename'] == filename]['region_count'].values[0]
    
    # Manual counting (modify as needed)
    print(f"Image: {filename}")
    print("Enter the number of regions you counted (or 's' to skip): ")
    user_count = input()
    
    if user_count.lower() != 's':
      try:
        user_count = int(user_count)
      except ValueError:
        print("Invalid input. Please enter a number or 's' to skip.")
        continue
    
      # Compare counts
      if user_count == expected_count:
        print(f"Match! Expected count: {expected_count}, Your count: {user_count}")
      else:
        print(f"Mismatch! Expected count: {expected_count}, Your count: {user_count}")
  else:
    print(f"Image not found: {filename}")

# Close any open windows (if image display was enabled)
cv2.destroyAllWindows()

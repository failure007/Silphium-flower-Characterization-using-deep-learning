import os
import shutil

# Function to rename CSV files and move them to a new folder
def rename_csv_files(directory):
    # Create a new folder named "extra_name"
    extra_folder = os.path.join(directory, "extra_nametag.")
    os.makedirs(extra_folder, exist_ok=True)

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Construct the new filename with "xtra" appended
            new_filename = os.path.splitext(filename)[0] + "extra.csv"
            # Construct the paths
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(extra_folder, new_filename)
            # Rename the file and move it to the "extra" folder
            shutil.move(old_path, new_path)
            print(f"Renamed {filename} to {new_filename} and moved to 'extra_name' folder.")

# Directory containing the CSV files
directory = r"C:\Users\DheerajKumarJallipal\Downloads\Annotated images (3)\Annotated images"

# Call the function to rename CSV files and move them to a new folder
rename_csv_files(directory)

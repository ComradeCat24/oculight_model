import os
import random
import shutil


# Set the path to the captions file and the directory where the images are located
datset_dir = 'dataset'
images_dir = os.path.join(datset_dir, "images")
captions_path = os.path.join(datset_dir, "captions.txt")

# Set the number of captions you want to select
num_captions = 30

# Read the captions file into a list
with open(captions_path, "r") as f:
    captions = f.readlines()

# Create a list of image names from the captions
image_names = list(set([line.split(",")[0] for line in captions]))

# Set the random seed
random.seed(42)

# Shuffle the list of image names
random.shuffle(image_names)

# Create a new directory to store the selected images and captions
subset_dir = "subset_data"
subset_images_dir = os.path.join(subset_dir, "images")
subset_captions_path = os.path.join(subset_dir, "captions.txt")

# Delete the subset_data directory and its contents
if os.path.exists(subset_dir):
    shutil.rmtree(subset_dir, ignore_errors=True)

# Create a new subset_data directory and its subdirectory images
os.makedirs(subset_dir)
os.makedirs(subset_images_dir)

# Select the first num_captions images from the shuffled list and write the captions to the subset captions file
selected_image_names = image_names[:num_captions]
with open(subset_captions_path, "w") as f:
    f.write("image_name,comment_number,comment\n")
    for image_name in selected_image_names:
        image_captions = [
            line for line in captions if line.startswith(image_name)]
        for line in image_captions:
            f.write(line)
        src_path = os.path.join(images_dir, image_name)
        dest_path = os.path.join(subset_images_dir, image_name)
        os.system(f"cp {src_path} {dest_path}")

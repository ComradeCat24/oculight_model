import os
import random

# Get the absolute path of the current working directory
cwd = os.path.abspath(os.getcwd())

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))

# Set the path to the captions file and the directory where the images are located
images_dir = f"{parent_dir}/dataset/flickr30k_images"
captions_path = f"{parent_dir}/dataset/captions.txt"

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
subset_dir = f"{parent_dir}/subset_dataset"
selected_images_dir = os.path.join(subset_dir, "selected_images")
subset_captions_path = os.path.join(subset_dir, "subset_captions.txt")
if not os.path.exists(selected_images_dir):
    os.makedirs(selected_images_dir)

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
        dest_path = os.path.join(selected_images_dir, image_name)
        os.system(f"cp {src_path} {dest_path}")

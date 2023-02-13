import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(caption_file, test_size=0.2, validation_size=0.2):
    if not os.path.isfile(caption_file):
        raise FileNotFoundError(f"{caption_file} not found.")

    # Extract the base file name
    base_file_name = os.path.splitext(os.path.basename(caption_file))[0]

    # Create a new directory to store the selected images and captions
    subset_dir = os.path.join(os.getcwd(), "subset_dataset")
    splits_dir = os.path.join(subset_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    # Generate file names
    train_file = os.path.join(splits_dir, f"{base_file_name}.train.txt")
    validation_file = os.path.join(
        splits_dir, f"{base_file_name}.validation.txt")
    test_file = os.path.join(splits_dir, f"{base_file_name}.test.txt")

    # Load data
    data = pd.read_csv('descriptions.txt', delimiter=',',  header=None, names=[
                       'image_name', 'comment_number', 'comment'])
    # skiprows=[0],
    data = data.drop_duplicates(subset=["image_name"], keep='first')
    print(data)

    # Split data into train, validation, and test sets
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=42)
    train_data, validation_data = train_test_split(
        train_data, test_size=validation_size, random_state=42)

    # Save data in separate files
    for filename, data in [(train_file, train_data), (validation_file, validation_data), (test_file, test_data)]:
        with open(filename, 'w') as file:
            for _, row in data.iterrows():
                file.write(row['image_name'])
                file.write('\n')


split_data('subset_dataset/subset_captions.txt')

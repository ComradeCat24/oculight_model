import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def split_data(caption_file, test_size=0.2, validation_size=0.2):
    if not os.path.isfile(caption_file):
        raise FileNotFoundError(f"{caption_file} not found.")
    # Extracting the base file name
    base_file_name = os.path.splitext(os.path.basename(caption_file))[0]

    # Generating file names
    train_file = f"dataset/splits/{base_file_name}.train.txt"
    validation_file = f"dataset/splits/{base_file_name}.validation.txt"
    test_file = f"dataset/splits/{base_file_name}.test.txt"

    data = pd.read_csv(caption_file, delimiter='\t')

    # Splitting the data into train, validation and test sets
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=42)
    train_data, validation_data = train_test_split(
        train_data, test_size=validation_size, random_state=42)

    # Saving the data in separate files
    for filename, data in tqdm([(train_file, train_data), (validation_file, validation_data), (test_file, test_data)], desc="Saving files"):
        with open(filename, 'w') as file:
            for _, row in data.iterrows():
                file.write(
                    row['image_name,comment_number,comment'].split(',')[0])
                file.write('\n')


split_data('dataset/captions.txt')

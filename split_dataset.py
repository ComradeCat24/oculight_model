# fmt: off
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from sklearn.model_selection import train_test_split
# fmt: on


def split_data(test_size=0.2):
    # Create a new directory to store the selected images and captions
    splits_dir = os.environ.get('SPLITS_PATH')
    os.makedirs(splits_dir, exist_ok=True)

    # Generate file names
    train_file = os.path.join(splits_dir, "captions.train.txt")
    test_file = os.path.join(splits_dir, "captions.test.txt")

    # Load data
    descriptions_file = os.environ.get('CLEANED_DESCRIPTIONS_FILE')
    data = pd.read_csv(descriptions_file, delimiter=',',  header=None, names=[
                       'image_name', 'comment_number', 'comment'])
    # skiprows=[0],
    data = data.drop_duplicates(subset=["image_name"], keep='first')
    # print(data)

    # Split data into train, validation, and test sets
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=42)

    # Save data in separate files
    for filename, data in [(train_file, train_data), (test_file, test_data)]:
        with open(filename, 'w') as file:
            for _, row in data.iterrows():
                file.write(row['image_name'])
                file.write('\n')


split_data()

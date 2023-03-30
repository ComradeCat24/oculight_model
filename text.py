# fmt: off
import os
import string
import pickle
from dotenv import load_dotenv
load_dotenv()
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
# fmt: on


def load_doc(filename):
    """
    Loads a text file and returns its content as a string.
    """
    try:
        with open(filename, 'r', encoding='utf-8-sig') as file:
            text = file.read()
    except IOError as e:
        print(e)
        print("Error occured while reading the file")
        return None
    return text


def load_descriptions(doc):
    """
    Loads a document and returns a mapping of image IDs to their corresponding
    descriptions.
    """
    mapping = defaultdict()
    for i, line in enumerate(doc.split('\n')):
        # skip the first iteration (removing header)
        if i == 0:
            continue
        tokens = line.split(',')
        if len(tokens) < 3:
            continue
        image_id, image_desc_id, *image_desc = tokens
        # image_id = image_id.split('.')[0] # w/ remove extension from image
        image_desc_id = image_desc_id.strip()  # w/o remove extension from image
        image_desc = ', '.join(image_desc).strip()
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append((image_desc_id, image_desc))
    return mapping


def clean_descriptions(descriptions):
    """
    Cleans the descriptions by removing punctuation, lowercasing all words, and
    removing words that are not alphabetic or have a length less than 2.
    """
    table = str.maketrans('', '', string.punctuation)
    for image_name, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc_id, desc = desc_list[i][0], desc_list[i][1]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = (desc_id, ' '.join(desc))
    return descriptions


def to_vocabulary(descriptions):
    """
    Extracts a set of all unique words from the descriptions.
    """
    all_desc = set()
    for img_id in descriptions:
        for desc in descriptions[img_id]:
            all_desc.update(desc[1].split())
    return all_desc


def save_descriptions(descriptions, filename):
    """
    Saves the descriptions to a text file in the format "image_name, comment_number, comment".
    """
    lines = list()
    for img_id in descriptions.keys():
        img_descs = descriptions[img_id]
        for desc in img_descs:
            lines.append(img_id + ', ' + desc[0] + ', ' + desc[1])
    data = '\n'.join(lines)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(data)


# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = []
    for key, values in descriptions.items():
        all_desc.extend(values)
    return all_desc


# load clean descriptions into memory
def load_clean_descriptions(filename):
    # load document
    with open(filename, "r", encoding='utf-8') as file:
        doc = file.readlines()

    descriptions = defaultdict(list)

    for line in doc:
        # split line by white space
        image_id, image_desc_id, image_desc = line.split(" ")[0].split(".")[0], line.split(" ")[
            1], " ".join(line.split(" ")[2:])

        # wrap description in tokens
        desc = 'startseq ' + image_desc.strip() + ' endseq'

        # store
        descriptions[image_id].append(desc)

    return descriptions


def create_tokenizer(descriptions, filename, num_words=None):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer(
        num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(lines)

    # save the tokenizer to a file
    with open(filename, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer


caption_directory = os.environ.get('CAPTION_DIRECTORY_PATH')
doc = load_doc(caption_directory)
descriptions = load_descriptions(doc)
descriptions = clean_descriptions(descriptions)

print(f'Descriptions: {len(descriptions)}')
vocabulary = to_vocabulary(descriptions)
print(f'Vocabulary Size: {len(vocabulary)}')

clean_descriptions_file = os.environ.get('CLEANED_DESCRIPTIONS_FILE')
save_descriptions(descriptions, clean_descriptions_file)

load_desc = load_clean_descriptions(clean_descriptions_file)

# prepare tokenizer
tokenized_file = os.environ.get('TOKENIZED_DATA_FILE')
create_tokenizer(load_desc, tokenized_file)

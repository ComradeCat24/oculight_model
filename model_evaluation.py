# fmt: off
import random
import os
import pickle
import textwrap
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import matplotlib.pyplot as plt
from collections import defaultdict
from keras.models import load_model
from keras.utils import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
# fmt: on


# load a pre-defined list of photo identifiers
def load_set(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        # process line by line
        lines = file.readlines()

    # get the image identifier
    dataset = [os.path.splitext(line.strip())[0] for line in lines]
    return dataset


# load clean descriptions into memory
def load_clean_descriptions(filename, dataset=None):
    # load document
    with open(filename, "r", encoding='utf-8') as file:
        doc = file.readlines()

    descriptions = defaultdict(list)

    for line in doc:
        # split line by white space
        image_id, image_desc = line.split(" ")[0].split(
            ".")[0], " ".join(line.split(" ")[2:])

        # if dataset is not empty or if the image is in the dataset, store the description
        if not dataset or image_id in dataset:

            # wrap description in tokens
            desc = 'startseq ' + image_desc.strip() + ' endseq'

            # store
            descriptions[image_id].append(desc)

    return descriptions


# load photo features
def load_photo_features(filename, dataset):
    try:
        with open(filename, 'rb') as f:
            all_features = pickle.load(f)
    except FileNotFoundError:
        print(f"{filename} not found.")
        return None

    features = {}
    for k, v in all_features.items():
        key = k.split('/')[-1]
        if key in dataset:
            features[key] = v

    return features


# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = []
    for key, values in descriptions.items():
        all_desc.extend(values)
    return all_desc


# load the tokenizer to a file
def load_tokenizer(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"{filename} not found.")
        return None


# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(map(lambda x: len(x.split()), lines))


def word_for_id(word_id, tokenizer, default='Unknown'):
    if word_id < 1 or word_id > len(tokenizer.word_index):
        raise ValueError(
            f"word_id should be between 1 and {len(tokenizer.word_index)}")
    return tokenizer.index_word.get(word_id, default)


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # initialize empty list to keep track of predicted words
    predicted_words = []
    i = 0
    photo = photo.reshape(1, photo.shape[0])
    # use while loop
    while i < max_length:
        try:
            # integer encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            # pad input
            sequence = pad_sequences([sequence], maxlen=max_length)
            # predict next word
            yhat = model.predict(
                [np.array(photo), np.array(sequence)], verbose=0)
            # print(np.array(yhat))
            # convert probability to integer
            yhat = np.argmax(yhat)
            # map integer to word
            word = word_for_id(yhat, tokenizer)
            # stop if we cannot map the word
            if word is None:
                break
            # append as input for generating the next word
            in_text += ' ' + word
            # keep track of predicted words
            predicted_words.append(word)
            # stop if we predict the end of the sequence
            if word == 'endseq':
                break
            i += 1
        except Exception as e:
            print(str(e))
            print('ERROR')
            return 'Error Generating Description'
    # remove 'startseq' from the predicted words
    predicted_words = predicted_words[1:]
    # join the predicted words to form a string
    final_text = " ".join(predicted_words)
    # remove 'endseq' from the final string
    final_text = final_text.replace('endseq', '')
    return final_text


def image_caption_plot(actual, predicted):

    # Randomly select 4 key-value pairs
    keys = list(actual.keys())
    random.shuffle(keys)
    keys = keys[:6]

    images_dir = os.environ.get('IMAGE_DIRECTORY_PATH')

    # Create a grid of subplots with 3 rows and 2 columns
    fig, axs = plt.subplots(3, 2, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    # Loop through each subplot and add an image and two captions
    for k, key in enumerate(keys):
        # Load the image and display it on the subplot
        image_file = f"{images_dir}/{key}.jpg"
        image = plt.imread(image_file)

        # Calculate the row and column index for the subplot
        row = k // 2
        col = k % 2

        wrapped_caption = textwrap.wrap(predicted[key], width=40)

        # Display the image and predicted caption in the subplot
        axs[row, col].imshow(image)
        axs[row, col].text(0.5, -0.1, '\n'.join(wrapped_caption), ha='center',
                           va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           transform=axs[row, col].transAxes, fontsize=14)
        # axs[row, col].set_title(predicted[key], fontsize=14)

    # Save the plot to a file
    evaluation_plot = os.environ.get('EVALUATION_PLOT_FILE')
    fig.savefig(evaluation_plot, bbox_inches="tight")


def calculate_bleu_scores(model, descriptions, photos, tokenizer, max_length):
    """
    Calculates BLEU scores for the model's generated descriptions.
    """
    list_of_references, hypotheses = [], []
    actual, predicted = defaultdict(), defaultdict()
    for i, (key, desc_list) in enumerate(descriptions.items()):
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        actual[key] = desc_list
        predicted[key] = yhat
        hypotheses.append(yhat.split())
        references = []
        for i in desc_list:
            words = i.split()
            references.append(words)
        list_of_references.append(references)

    bleu_scores = corpus_bleu(list_of_references, hypotheses)

    image_caption_plot(actual, predicted)

    return bleu_scores


# test dataset

# load training dataset

test_filename = os.environ.get('TESTING_SET')

test = load_set(test_filename)
print('Dataset: %d' % len(test))

# descriptions
descriptions_file = os.environ.get('CLEANED_DESCRIPTIONS_FILE')
test_descriptions = load_clean_descriptions(descriptions_file, test)
print('Descriptions: test=%d' % len(test_descriptions))

# prepare tokenizer
tokenized_file = os.environ.get('CAPTION_TOKENIZERS_FILE')
tokenizer = load_tokenizer(tokenized_file)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# determine the maximum sequence length
max_length = max_length(load_clean_descriptions(descriptions_file))
print('Description Length: %d' % max_length)

# photo features
pickle_file = os.environ.get('IMAGE_FEATURES_FILE')
test_features = load_photo_features(pickle_file, test)
print('Photos: test=%d' % len(test_features))

# load the model
model_file = os.environ.get('SAVED_MODEL_FILE')
model = load_model(model_file)

# evaluate model
score = calculate_bleu_scores(
    model, test_descriptions, test_features, tokenizer, max_length)
print('[info] CORPUS-LEVEL BLEU SCORE: {:.10f}'.format(score))

import os
import pickle
import random
import numpy as np
from collections import defaultdict
from keras.models import load_model
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.translate.bleu_score import corpus_bleu
from keras.utils import to_categorical, plot_model, pad_sequences


# load doc into memory
def load_doc(filename):
    if os.path.isfile(filename):
        try:
            with open(filename, 'rb', encoding='utf-8') as file:
                return file.read()
        except IOError as e:
            print(f"Error Occured: {e}")
    else:
        print(f"{filename} not found.")


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
        image_id, image_desc_id, image_desc = line.split(" ")[0].split(".")[0], line.split(" ")[
            1], " ".join(line.split(" ")[2:])

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
    print("---1---")
    # seed the generation process
    in_text = 'startseq'
    # initialize empty list to keep track of predicted words
    predicted_words = []
    i = 0
    # use while loop
    while i < max_length:
        try:
            print(in_text)
            print("---2---")
            # integer encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            print(sequence)
            print("---3---")
            # pad input
            sequence = pad_sequences([sequence], maxlen=max_length)
            print(sequence)
            print("---4---")
            print(f"shape of photo is {photo.shape}")
            print(f"shape of sequence is {sequence.shape}")
            print("---6---")
            # predict next word
            yhat = model.predict(np.array([photo, sequence]), verbose=1)
            print("---7---")
            # convert probability to integer
            yhat = np.argmax(yhat)
            print("---8---")
            # map integer to word
            # word = tokenizer.index_word[yhat]
            word = word_for_id(yhat)
            print("---9---")
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
            print("---100---")
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
    return final_text, predicted_words


def calculate_bleu_scores(model, descriptions, photos, tokenizer, max_length):
    """
    Calculates BLEU scores for the model's generated descriptions.
    """
    bleu_weights = [(1.0, 0, 0, 0), (0.5, 0.5, 0, 0),
                    (0.3, 0.3, 0.3, 0), (0.25, 0.25, 0.25, 0.25)]
    actual, predicted = [], []
    for i, (key, desc_list) in enumerate(descriptions.items()):
        # yhat = generate_desc(model, tokenizer, desc_list, photos[key][0], max_length)
        print(key)
        print(desc_list)
        print("START")
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        print("END")
    #     actual.append([d.split() for d in desc_list])
    #     predicted.append(yhat.split())
    #     print(f"actual: {actual} \npredicted: {predicted}")
    # bleu_scores = {f"BLEU-{i+1}": corpus_bleu(actual, predicted, weights=w)
    #                for i, w in enumerate(bleu_weights)}
    # print(bleu_scores)
    # return bleu_scores
    return -1


# test dataset

# load training dataset
test_filename = 'subset_dataset/splits/subset_captions.test.txt'

test = load_set(test_filename)
print('Dataset: %d' % len(test))

# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))

# prepare tokenizer
tokenizer = load_tokenizer('tokenizer.pkl')
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# determine the maximum sequence length
max_length = max_length(load_clean_descriptions('descriptions.txt'))
print('Description Length: %d' % max_length)

# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: train=%d' % len(test_features))


# load the model
filename = 'model_checkpoints/new_model.h5'
model = load_model(filename)
# evaluate model
calculate_bleu_scores(model, test_descriptions,
                      test_features, tokenizer, max_length)

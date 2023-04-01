# fmt: off
import os
import sys
import pickle
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from collections import defaultdict
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
# from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import pad_sequences, load_img, img_to_array
# fmt: on


# load clean descriptions into memory
def load_clean_descriptions(filename):
    # load document
    with open(filename, "r", encoding='utf-8') as file:
        doc = file.readlines()

    descriptions = defaultdict(list)

    for line in doc:
        # split line by white space
        image_id, _, image_desc = line.split(" ")[0].split(".")[0], line.split(" ")[
            1], " ".join(line.split(" ")[2:])

        # wrap description in tokens
        desc = 'startseq ' + image_desc.strip() + ' endseq'

        # store
        descriptions[image_id].append(desc)

    return descriptions


# load photo features
def extract_photo_features(filename):
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # OR
    # model = MobileNetV2(weights='imagenet', include_top=False)

    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape(
        (1, image.shape[0], image.shape[1], image.shape[2]))
    feature = model.predict(datagen.flow(
        image), verbose=0).flatten()

    return np.array(feature)

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


# load tokenizer
tokenized_file = os.environ.get('CAPTION_TOKENIZERS_FILE')
tokenizer = load_tokenizer(tokenized_file)

# determine the maximum sequence length
descriptions_file = os.environ.get('CLEANED_DESCRIPTIONS_FILE')
max_length = max_length(load_clean_descriptions(descriptions_file))

# load the model
model_file = os.environ.get('SAVED_MODEL_FILE')
model = load_model(model_file)

# Loop through all command line arguments
for i in range(1, len(sys.argv)):
    # photo feature
    image_filename = sys.argv[i]
    photo_feature = extract_photo_features(image_filename)

    # generate description
    desc = generate_desc(model, tokenizer, photo_feature, max_length)
    # print(f'{image_filename}: {desc}')
    print(desc)

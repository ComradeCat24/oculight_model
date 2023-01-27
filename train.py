import json
import os
from collections import defaultdict
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Add
from keras.callbacks import ModelCheckpoint


# load doc into memory
def load_doc(filename):
    if os.path.isfile(filename):
        try:
            with open(filename, 'rb') as file:
                return file.read()
        except IOError as e:
            print(f"Error Occured: {e}")
    else:
        print(f"{filename} not found.")


# load a pre-defined list of photo identifiers
def load_set(filename):
    with open(filename, 'r') as file:
        # process line by line
        lines = file.readlines()

    # get the image identifier
    dataset = {os.path.splitext(line.strip())[0] for line in lines}
    return dataset


# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    # load document
    with open(filename, "r") as file:
        doc = file.readlines()

    descriptions = defaultdict(list)

    for line in doc:
        # split line by white space
        image_id, image_desc = line.split(
            " ")[0], " ".join(line.split(" ")[1:])

        # skip images not in the set
        if image_id in dataset:
            # wrap description in tokens
            desc = 'startseq ' + image_desc.strip() + ' endseq'

            # store
            descriptions[image_id].append(desc)

    return descriptions


# load photo features
def load_photo_features(filename, dataset):
    try:
        with open(filename, 'r') as file:
            all_features = json.load(file)
    except FileNotFoundError:
        print(f"{filename} not found.")
        return None

    features = filter(lambda x: x[0] in dataset, all_features.items())
    return dict(features)


# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = []
    for key, values in descriptions.items():
        all_desc.extend(values)
    return all_desc


# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions, num_words=None):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer(
        num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(lines)
    return tokenizer


# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(map(lambda x: len(x.split()), lines))


# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)


# define the captioning model
def define_model(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = Add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            # retrieve the photo feature
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(
                tokenizer, max_length, desc_list, photo, vocab_size)
            yield [in_img, in_seq], out_word


# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'

train = load_set(filename)
print('Dataset: %d' % len(train))

# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))

# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# define the model
model = define_model(vocab_size, max_length)
# train the model, run epochs manually and save after each epoch
epochs = 20
steps = len(train_descriptions)
for i in range(epochs):
    # create the data generator
    generator = data_generator(
        train_descriptions, train_features, tokenizer, max_length, vocab_size)
    # fit for one epoch
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    # save model
    model.save('model_' + str(i) + '.h5')

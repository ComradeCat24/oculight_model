import os
import random
import numpy as np
import pickle
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, plot_model, pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, TimeDistributed, GlobalMaxPooling1D, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2


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
def load_clean_descriptions(filename, dataset):
    # load document
    with open(filename, "r", encoding='utf-8') as file:
        doc = file.readlines()

    descriptions = defaultdict(list)

    for line in doc:
        # split line by white space
        image_id, image_desc_id, image_desc = line.split(" ")[0].split(".")[0], line.split(" ")[
            1], " ".join(line.split(" ")[2:])

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
        with open(filename, 'rb') as f:
            all_features = pickle.load(f)
    except FileNotFoundError:
        print(f"{filename} not found.")
        return None

    # Extract only the keys in the dataset and split the keys to get the image IDs
    # features = {k.split('/')[-1]: v for k, v in all_features.items() if k.split('/')[-1] in dataset}

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


# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions, num_words=None):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer(
        num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(lines)

    # save the tokenizer to a file
    with open('tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer


# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(map(lambda x: len(x.split()), lines))


# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    X1, X2, y = [], [], []
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
            X1.append(photo[0])
            X2.append(in_seq)
            y.append(out_seq)

    return np.array(X1), np.array(X2), np.array(y)


# define the captioning model
def define_model(vocab_size, max_length):

    inputs1 = Input(shape=(7, 1280))
    fe1 = TimeDistributed(Dropout(0.5))(inputs1)
    fe2 = TimeDistributed(Dense(256, activation='relu',
                          kernel_regularizer=l2(0.01)))(fe1)
    fe3 = GlobalMaxPooling1D()(fe2)

    inputs2 = Input(shape=(max_length))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256, return_sequences=True)(se2)
    se4 = GlobalMaxPooling1D()(se3)

    decoder1 = Concatenate()([fe3, se4])
    decoder2 = Dense(256, activation='relu',
                     kernel_regularizer=l2(0.01))(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # summarize model
    model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)

    return model


# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length, vocab_size, n_iterations=None):
    keys = list(descriptions.keys())
    i = 0
    while True:
        if n_iterations and i >= n_iterations:
            break
        random.shuffle(keys)
        for key in keys:
            print(key)
            try:
                # retrieve the photo feature
                photo = photos[key][0]
                in_img, in_seq, out_word = create_sequences(
                    tokenizer, max_length, descriptions[key], photo, vocab_size)
                # print(f"in_img has shape of {in_img.shape}")
                # print(f"in_img has {in_img}")
                # print(f"in_seq has shape of {in_seq.shape}")
                # print(f"in_seq has {in_seq}")
                # print(f"out_word has shape of {out_word.shape}")
                # print(f"out_word has {out_word}")
                yield [in_img, in_seq], out_word
                i += 1
            except Exception as e:
                print(f"Error in iteration {i} with key {key} : {e}")


# train dataset

# load training dataset
train_filename = 'subset_dataset/splits/subset_captions.train.txt'

train = load_set(train_filename)
print('Dataset: %d' % len(train))

# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))


# fit model

# define the model
model = define_model(vocab_size, max_length)

# train the model, run epochs manually and save after each epoch
epochs = 15
steps_per_epoch = len(train_descriptions)

# create the data generator
train_generator = data_generator(
    train_descriptions, train_features, tokenizer, max_length, vocab_size)

# create callbacks list
callbacks_list = [
    EarlyStopping(monitor='loss', patience=3),
    ModelCheckpoint(
        filepath='model_checkpoints/model.h5', save_best_only=True, monitor='loss')
    # ModelCheckpoint(filepath='model_checkpoints/model_{epoch:02d}.h5', save_best_only=True, monitor='loss')
]

# fit the model
model.fit_generator(
    generator=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks_list,
)

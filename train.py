import os
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, plot_model, pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add, TimeDistributed, GlobalMaxPooling1D, Concatenate, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import warnings
from keras.regularizers import l2

warnings.filterwarnings('ignore')


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
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0].flatten()
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)

    # print(np.array(X1).shape)
    # print(np.array(X2).shape)
    # print(np.array(y).shape)
    return np.array(X1), np.array(X2), np.array(y)


# define the captioning model
def define_model(feature_shape, vocab_size, max_length):

    input_image = Input(shape=(feature_shape))
    droplayer1 = Dropout(0.5)(input_image)
    bn = BatchNormalization()(droplayer1)
    fimage1 = Dense(256, activation='relu', name="ImageFeature1",
                    kernel_regularizer=l2(0.01))(bn)
    droplayer1 = Dropout(0.5)(fimage1)
    bn = BatchNormalization()(droplayer1)
    fimage1 = Dense(256, activation='relu', name="ImageFeature3",
                    kernel_regularizer=l2(0.01))(bn)
    droplayer2 = Dropout(0.5)(fimage1)
    bn = BatchNormalization()(droplayer2)
    fimage2 = Dense(256, activation='relu')(bn)

    # sequence model
    input_txt = Input(shape=(max_length))
    ftxt = Embedding(vocab_size, 64, mask_zero=True)(input_txt)
    droplayer_ = Dropout(0.5)(ftxt)
    bn = BatchNormalization()(droplayer_)
    ftxt = LSTM(256, name="CaptionFeature")(bn)

    # combined model for decoder
    decoder = Add()([ftxt, fimage2])
    decoder = Dense(256, activation='relu')(decoder)
    droplayer1 = Dropout(0.5)(decoder)
    bn = BatchNormalization()(droplayer1)
    output = Dense(vocab_size, activation='softmax')(bn)

    model = Model(inputs=[input_image, input_txt], outputs=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # summarize model
    model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)

    return model


# data generator, intended to be used in a call to model.fit()
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
                photo = photos[key]
                in_img, in_seq, out_word = create_sequences(
                    tokenizer, max_length, descriptions[key], photo, vocab_size)
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
tokenizer = load_tokenizer('tokenizer.pkl')
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# determine the maximum sequence length
max_length = max_length(load_clean_descriptions('descriptions.txt'))
print('Description Length: %d' % max_length)

# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))
# Get the key-value pair for the first item in the dictionary
_, image_feature = next(iter(train_features.items()))
feature_shape = np.array(image_feature).shape
print('Photos: shape=%d' % feature_shape)

# fit model

# define the model
model = define_model(feature_shape, vocab_size, max_length)

# train the model, run epochs manually and save after each epoch
epochs = 20
steps_per_epoch = len(train_descriptions)

# create the data generator
train_generator = data_generator(
    train_descriptions, train_features, tokenizer, max_length, vocab_size)

# create callbacks list
callbacks_list = [
    # for loss
    # EarlyStopping(monitor='loss', mode='min', patience=3),
    # ModelCheckpoint(filepath='model_checkpoints/model_{epoch:02d}.h5', save_best_only=True, monitor='loss', mode='min')

    # for accuracy
    EarlyStopping(monitor='accuracy', mode='max', patience=3),
    ModelCheckpoint(filepath='model_checkpoints/model.h5',
                    save_best_only=True, monitor='accuracy', mode='max')
]

# fit the model
model.fit(
    x=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    verbose=1,
    # use_multiprocessing=True,
    # workers=num_workers, # specify the number of worker processes to use
    # max_queue_size=max_queue_size, # specify the maximum size of the generator queue
    callbacks=callbacks_list,
)

print(model.history.history.keys())

# Get the training loss and accuracy
train_loss = model.history.history['loss']
train_acc = model.history.history['accuracy']

# Get the number of epochs
epochs = range(1, len(train_loss) + 1)

# Calculate the mean training loss and accuracy
mean_loss = np.mean(train_loss)
mean_acc = np.mean(train_acc)

# Plot training loss and accuracy
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.plot(epochs, train_loss, 'bo', label='Training loss')
ax1.axhline(mean_loss, color='r', label=f'Mean loss: {mean_loss:.2f}')
ax1.set_title('Training loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(epochs, train_acc, 'bo', label='Training accuracy')
ax2.axhline(mean_acc, color='m', label=f'Mean accuracy: {mean_acc:.2f}')
ax2.set_title('Training accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig('training_plots.png')

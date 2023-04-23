# fmt: off
import os
import glob
import random
import pickle
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import matplotlib.pyplot as plt
from collections import defaultdict
from keras.regularizers import l2
from keras.utils import to_categorical, plot_model, pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
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
def define_model(image_shape, vocab_size, max_caption_length):

    # Image model
    input_image = Input(shape=image_shape, name="image_input")
    dropout_layer = Dropout(0.5)(input_image)
    bn_layer = BatchNormalization()(dropout_layer)
    dense_layer1 = Dense(256, activation='relu',
                         kernel_regularizer=l2(0.01))(bn_layer)
    dropout_layer = Dropout(0.5)(dense_layer1)
    bn_layer = BatchNormalization()(dropout_layer)
    dense_layer2 = Dense(256, activation='relu',
                         kernel_regularizer=l2(0.01))(bn_layer)
    dropout_layer = Dropout(0.5)(dense_layer2)
    bn_layer = BatchNormalization()(dropout_layer)
    image_features = Dense(256, activation='relu',
                           name="image_features")(bn_layer)

    # Caption model
    input_caption = Input(shape=(max_caption_length,), name="caption_input")
    embedding_layer = Embedding(vocab_size, 64, mask_zero=True)(input_caption)
    dropout_layer = Dropout(0.5)(embedding_layer)
    bn_layer = BatchNormalization()(dropout_layer)
    caption_features = LSTM(256, name="caption_features")(bn_layer)

    # Combined model
    decoder = Add()([image_features, caption_features])
    dense_layer1 = Dense(256, activation='relu')(decoder)
    dropout_layer = Dropout(0.5)(dense_layer1)
    bn_layer = BatchNormalization()(dropout_layer)
    output = Dense(vocab_size, activation='softmax', name="output")(bn_layer)

    # Create and compile the model
    model = Model(inputs=[input_image, input_caption], outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Summarize the model
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

train_filename = os.environ.get('TRANING_SET')

train = load_set(train_filename)
print('Dataset: %d' % len(train))

# descriptions
descriptions_file = os.environ.get('CLEANED_DESCRIPTIONS_FILE')
train_descriptions = load_clean_descriptions(descriptions_file, train)
print('Descriptions: train=%d' % len(train_descriptions))

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
train_features = load_photo_features(pickle_file, train)
print('Photos: train=%d' % len(train_features))
# Get the key-value pair for the first item in the dictionary
_, image_feature = next(iter(train_features.items()))
feature_shape = np.array(image_feature).shape
print('Photos: shape=%d' % feature_shape)

# fit model

# define the model
model = define_model(feature_shape, vocab_size, max_length)

# train the model, run epochs manually and save after each epoch
epochs = int(os.environ.get('TRAINING_EPOCHS'))
initial_epoch = 0
steps_per_epoch = len(train_descriptions)

# create the data generator
train_generator = data_generator(
    train_descriptions, train_features, tokenizer, max_length, vocab_size)

model_checkpoints_dir = os.environ.get('CHECKPOINT_DIR_PATH')
os.makedirs(model_checkpoints_dir, exist_ok=True)


def delete_old_checkpoints(model_checkpoints_dir, keep=2):
    # Get a list of all the checkpoint files in the directory
    checkpoint_files = glob.glob(f'{model_checkpoints_dir}/*.hdf5')

    if checkpoint_files:
        # Sort the checkpoint files by epoch number
        checkpoint_files = sorted(
            checkpoint_files, key=lambda x: int(x.split('.')[-2]))

        # Keep only the most recent 'keep' number of files
        if len(checkpoint_files) > keep:
            checkpoint_files_to_delete = checkpoint_files[:-keep]
            for checkpoint_file in checkpoint_files_to_delete:
                os.remove(checkpoint_file)


# create callbacks list
callbacks_list = [
    # for accuracy
    EarlyStopping(monitor='accuracy', mode='max',
                  patience=5, restore_best_weights=True),
    ModelCheckpoint(filepath=os.path.join(model_checkpoints_dir,
                    'weights.{epoch:02d}-{accuracy:.2f}.hdf5'), save_best_only=True, save_weights_only=True, monitor='accuracy', mode='max'),
    LambdaCallback(
        on_epoch_end=lambda epoch, logs: delete_old_checkpoints(
            model_checkpoints_dir, keep=2)
    )
]


# Get a list of all the checkpoint files in the directory
checkpoint_files = glob.glob(f'{model_checkpoints_dir}/*.hdf5')

if checkpoint_files:
    # Sort the checkpoint files by epoch number
    checkpoint_files = sorted(
        checkpoint_files, key=lambda x: int(x.split('.')[-2]))

    # Load the last checkpoint file in the sorted list (which has the highest epoch number)
    latest_checkpoint_file = checkpoint_files[-1]

    print("Found checkpoint file:", latest_checkpoint_file)

    # Extract the epoch number from the third-to-last part of the file name
    initial_epoch = int(latest_checkpoint_file.split('.')[-3].split('-')[0])

    model.load_weights(latest_checkpoint_file)

    # Print a message indicating that the training is resuming from a specific epoch
    print("Resuming training from epoch:", initial_epoch + 1)
else:
    # No checkpoint files found
    print("No checkpoint files found in 'model_checkpoints' directory.")


# fit the model
model.fit(
    x=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    initial_epoch=initial_epoch,
    verbose=1,
    # use_multiprocessing=True,
    # workers=num_workers, # specify the number of worker processes to use
    # max_queue_size=max_queue_size, # specify the maximum size of the generator queue
    callbacks=callbacks_list,
)

model_file = os.environ.get('SAVED_MODEL_FILE')
model.save(model_file)

# Get the training loss and accuracy
train_loss = model.history.history['loss']
train_acc = model.history.history['accuracy']

# Get the number of epochs
epochs = range(1, len(train_loss) + 1)

# Fit a polynomial regression line to the training loss and accuracy data
loss_coeffs = np.polyfit(epochs, train_loss, 1)
acc_coeffs = np.polyfit(epochs, train_acc, 1)
loss_fit = np.poly1d(loss_coeffs)
acc_fit = np.poly1d(acc_coeffs)

# Plot training loss and accuracy with best-fitting lines
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.plot(epochs, train_loss, 'bo', label='Training loss')
ax1.plot(epochs, loss_fit(epochs), 'r-', label='Best fit line')
ax1.set_title('Training loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(epochs, train_acc, 'bo', label='Training accuracy')
ax2.plot(epochs, acc_fit(epochs), 'r-', label='Best fit line')
ax2.set_title('Training accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
training_plot = os.environ.get('TRAINING_PLOT_FILE')
plt.savefig(training_plot)

# fmt: off
import os
import pickle 
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
from keras.models import Model
from collections import OrderedDict
from keras.utils import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
# from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# fmt: on


def extract_features(directory, batch_size=500):

    # model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # OR
    model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), include_top=False)

    # Freeze the weights of the MobileNetV2 layer
    for layer in model.layers:
        layer.trainable = False

    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    model.summary()

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Load existing features from pickle file
    pickle_file = os.environ.get('IMAGE_FEATURES_FILE')
    if os.path.exists(pickle_file):
        features = pickle.load(open(pickle_file, 'rb'))
    else:
        features = OrderedDict()

    tqdm.write("\n[info] STARTING EXTRACTING FEATURES...")
    for i, name in enumerate(tqdm(os.listdir(directory))):
        filename = os.path.join(directory, name)
        image_id = name.split(".")[0]

        # Check if image_id is already present in the features OrderedDict
        if image_id in features:
            # tqdm.write(f'[info] SKIPPING IMAGE {name} AS ITS FEATURES ARE ALREADY EXTRACTED.')
            continue

        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))  # (1, 224, 224, 3)
        feature = model.predict(datagen.flow(
            image), verbose=0)  # (1, 14, 14, 512)

        feature = feature.flatten()  # (100352)
        # OR
        # feature = feature.reshape((feature.shape[0], feature.shape[1] * feature.shape[2], feature.shape[3]))  # (1, 196, 512)

        features.update({image_id: feature})

        if i % batch_size == 0 or i == len(os.listdir(directory))-1:
            tqdm.write(
                f"\n[info] IMAGE FEATURES CHECKPOINT SAVING AT {len(features)}")
            # Save updated features to pickle file after processing every batch of images or after processing the last image
            pickle.dump(features, open(pickle_file, 'wb'))


directory = os.environ.get('IMAGE_DIRECTORY_PATH')
extract_features(directory)

tqdm.write("\n[info] IMAGE FEATURES FILE SAVED SUCCESSFULLY")

# fmt: off
import os
import numpy as np
import pickle 
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
from keras.models import Model
from collections import OrderedDict
from keras.utils import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# fmt: on


def extract_features(directory):

    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Load existing features from pickle file
    pickle_file = os.environ.get('IMAGE_FEATURES_FILE')
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            features = pickle.load(f)
    else:
        features = OrderedDict()

    print("\n[info] STARTING EXTRACTING FEATURES...")
    for name in tqdm(os.listdir(directory)):
        filename = directory + '/' + name
        image_id = name.split(".")[0]

        # Check if image_id is already present in the features OrderedDict
        if image_id in features:
            continue

        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))
        feature = model.predict(datagen.flow(
            image, batch_size=32), verbose=0).flatten()

        features.update({image_id: feature})

        # Save updated features to pickle file
        with open(pickle_file, 'wb') as f:
            pickle.dump(features, f)


directory = os.environ.get('IMAGE_DIRECTORY_PATH')
extract_features(directory)

print("\n[info] IMAGE FEATURES FILE SAVED SUCCESSFULLY")

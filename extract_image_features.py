# fmt: off
import os
import pickle 
from dotenv import load_dotenv
load_dotenv()
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
    model = MobileNetV2(weights='imagenet', input_shape=(
        224, 224, 3), include_top=False)
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    model.summary()

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Load existing features from pickle file
    pickle_file = os.environ.get('IMAGE_FEATURES_FILE')
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            features = pickle.load(f)
    else:
        features = OrderedDict()

    num_processed = 0
    for i, name in enumerate(os.listdir(directory)):
        filename = os.path.join(directory, name)
        image_id = name.split(".")[0]

        # Check if image_id is already present in the features OrderedDict
        if image_id in features:
            print(
                f'Skipping image {name} as its features are already extracted.')
            continue

        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)

        image = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))
        print(image.shape)
        feature = model.predict(datagen.flow(image), verbose=0)
        print(feature.shape)
        feature = feature.reshape(
            (feature.shape[0], feature.shape[1] * feature.shape[2], feature.shape[3]))
        print(feature.shape)

        print('>%s' % name)
        features.update({image_id: feature})
        num_processed += 1

        if num_processed % batch_size == 0 or i == len(os.listdir(directory))-1:
            # Save updated features to pickle file after processing every batch of images or after processing the last image
            with open(pickle_file, 'wb') as f:
                pickle.dump(features, f)
            print('Processed %d images, extracted features: %d' %
                  (num_processed, len(features)))

    print('Processed %d images, extracted features: %d' %
          (num_processed, len(features)))


directory = os.environ.get('IMAGE_DIRECTORY_PATH')
extract_features(directory)

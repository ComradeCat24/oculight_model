from os import listdir
import pickle
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

# Preparing Photo Data


def extract_features(directory):

    model = MobileNetV2(weights='imagenet', include_top=False)
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    print(model.summary())

    features = {}

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Generate batches of image data from the specified directory
    data_gen = datagen.flow_from_directory(
        directory, target_size=(150, 150), batch_size=32, class_mode=None)

    # Extract features from each batch
    for features_batch in model.predict(data_gen, verbose=1):
        for i in range(len(data_gen.filenames)):
            image_id = data_gen.filenames[i].split('.')[0]
            features[image_id] = features_batch[i]
            print('>%s' % data_gen.filenames[i])

    with open('features.pickle', 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return features


# directory = 'Flicker30k_Dataset'
directory = 'dataset'
features = extract_features(directory)

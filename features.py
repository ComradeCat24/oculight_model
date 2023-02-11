from os import listdir
from pickle import dump
# from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from keras.models import Model


def extract_features(directory):

    # model = VGG16(include_top=False, input_shape=(224, 224, 3))

    model = MobileNetV2(weights='imagenet', include_top=False)
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    model.summary()

    features = dict()
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    def feature_generator(directory):
        for name in listdir(directory):
            filename = directory + '/' + name
            image_id = name.split(".")[0]
            image = load_img(filename, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape(
                (1, image.shape[0], image.shape[1], image.shape[2]))
            feature = model.predict(datagen.flow(image, batch_size=32))
            print('>%s' % name)
            yield image_id, feature

    features = dict(feature_generator(directory))
    return features


directory = 'subset_dataset/selected_images'
features = extract_features(directory)
print(f'Features: ${features}')
print('Extracted Features: %d' % len(features))
dump(features, open('features.pkl', 'wb'))

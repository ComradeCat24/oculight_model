from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model

# Preparing Photo Data

# The code first loads the VGG16 model and removes the last layers so that it
# only outputs the features of the last fully connected layer. Then, it reads
# all image files in the specified directory, loads them, resizes them to (224,
# 224) pixels, preprocesses them, and uses the VGG16 model to extract features
# from each image.

# It then associates the image id with the extracted features from the image and
# store in features dictionary.

def extract_features(directory):
	model = VGG16()
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	print(model.summary())
	features = {}

	for name in listdir(directory):
		filename = directory + '/' + name
		# print(filename)
		image = load_img(filename, target_size=(224, 224))
		image = img_to_array(image)
		image = image.reshape(
			(1, image.shape[0], image.shape[1], image.shape[2])
		)
		image = preprocess_input(image)
		feature = model.predict(image, verbose=0)
		image_id = name.split('.')[0]
		features[image_id] = feature
		print('>%s' % name)

	return features


directory = 'Flicker30k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
dump(features, open('features.pkl', 'wb'))

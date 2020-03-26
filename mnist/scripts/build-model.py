#!/usr/bin/env python3

import os

import tensorflow

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

DATA_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'data', 'mnist'))
MODEL_PATH = os.path.join(DATA_DIR, 'neuralclassifier_model.h5')
FEATURES_PATH = os.path.join(DATA_DIR, 'neuralclassifier_features.txt')
LABELS_PATH = os.path.join(DATA_DIR, 'neuralclassifier_labels.txt')
PREDICTED_NUMBER_TARGETS_PATH = os.path.join(DATA_DIR, 'predictednumber_targets.txt')
PREDICTED_NUMBER_TRUTH_PATH = os.path.join(DATA_DIR, 'predictednumber_truth.txt')
NEURAL_CLASSIFIER_TARGETS_PATH = os.path.join(DATA_DIR, 'neuralclassifier_targets.txt')

# This is also the order of the labels in the output layer.
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

HIDDEN_LAYER_SIZE = 128
HIDDEN_LAYER_ACTIVATION = 'relu'

SIGNIFICANT_DIGITS = 4

# We don't care if our model if very accurate.
EPOCHS = 3

# Max: 10000
NUM_TEST_IMAGES = 1000

def normalizeImages(images):
    (numImages, width, height) = images.shape

    # Flatten out the images into a 1d array.
    images = images.reshape(numImages, width * height)

    # Normalize the greyscale intensity to [0,1].
    images = images / 255.0

    # Round so that the output is significantly smaller.
    images = images.round(SIGNIFICANT_DIGITS)

    return images

def loadData():
    mnist = tensorflow.keras.datasets.mnist
    (trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()

    trainImages = normalizeImages(trainImages)
    testImages = normalizeImages(testImages)

    testImages = testImages[0:NUM_TEST_IMAGES]
    testLabels = testLabels[0:NUM_TEST_IMAGES]

    return (trainImages, trainLabels), (testImages, testLabels)

def train(images, labels):
    inputSize = len(images[0])

    model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Dense(inputSize, input_shape=(inputSize,)),
        tensorflow.keras.layers.Dense(HIDDEN_LAYER_SIZE, activation = HIDDEN_LAYER_ACTIVATION),
        tensorflow.keras.layers.Dense(len(LABELS))
    ])

    model.compile(
        optimizer = 'adam',
        loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = ['accuracy']
    )

    model.fit(images, labels, epochs = EPOCHS)

    return model

def test(model, images, labels):
    loss, accuracy = model.evaluate(images, labels)
    return (loss, accuracy)

def writeFile(path, data):
    with open(path, 'w') as file:
        for row in data:
            file.write('\t'.join([str(item) for item in row]) + "\n")

def writeData(model, testImages, testLabels):
    model.save(MODEL_PATH)

    os.makedirs(DATA_DIR, exist_ok = True)

    features = []
    targets = []
    truths = []

    for i in range(len(testImages)):
        features.append([i] + list(testImages[i]))

        for j in range(len(LABELS)):
            targets.append([i, LABELS[j]])

            if (testLabels[i] == j):
                truths.append([i, LABELS[j]])

    writeFile(FEATURES_PATH, features)
    writeFile(PREDICTED_NUMBER_TARGETS_PATH, targets)
    writeFile(NEURAL_CLASSIFIER_TARGETS_PATH, targets)
    writeFile(PREDICTED_NUMBER_TRUTH_PATH, truths)

    writeFile(LABELS_PATH, LABELS)

def main():
    (trainImages, trainLabels), (testImages, testLabels) = loadData()
    inputSize = len(trainImages[0])

    model = train(trainImages, trainLabels)

    loss, accuracy = test(model, testImages, testLabels)
    print("Trained Model -- Loss: %f, Accuracy: %f" % (loss, accuracy))

    writeData(model, testImages, testLabels)

    pretrainedModel = tensorflow.keras.models.load_model(MODEL_PATH)

    loss, accuracy = test(pretrainedModel, testImages, testLabels)
    print("Pretrained Model -- Loss: %f, Accuracy: %f" % (loss, accuracy))

if (__name__ == '__main__'):
    main()

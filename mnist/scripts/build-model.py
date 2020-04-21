#!/usr/bin/env python3

import os

import numpy
import tensorflow

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

DATA_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'data', 'mnist'))
UNTRAINED_MODEL_PATH = os.path.join(DATA_DIR, 'neuralclassifier_model_untrained.h5')
TRAINED_MODEL_PATH = os.path.join(DATA_DIR, 'neuralclassifier_model_trained.h5')
FEATURES_PATH = os.path.join(DATA_DIR, 'neuralclassifier_features.txt')
LABELS_PATH = os.path.join(DATA_DIR, 'neuralclassifier_labels.txt')
NEURAL_CLASSIFIER_TARGETS_PATH = os.path.join(DATA_DIR, 'neuralclassifier_targets.txt')
PREDICTED_NUMBER_OBS_PATH = os.path.join(DATA_DIR, 'predictednumber_obs.txt')
PREDICTED_NUMBER_TARGETS_PATH = os.path.join(DATA_DIR, 'predictednumber_targets.txt')
PREDICTED_NUMBER_TRUTH_PATH = os.path.join(DATA_DIR, 'predictednumber_truth.txt')

# This is also the order of the labels in the output layer.
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

HIDDEN_LAYER_SIZE = 128
HIDDEN_ACTIVATION = 'relu'
OUTPUT_ACTIVATION = 'softmax'

SIGNIFICANT_DIGITS = 4

# We don't care if our model is very accurate.
EPOCHS = 3

# Max: 60000
NUM_TRAIN_IMAGES = 6000

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

    trainImages = trainImages[0:NUM_TRAIN_IMAGES]
    trainLabels = trainLabels[0:NUM_TRAIN_IMAGES]

    testImages = testImages[0:NUM_TEST_IMAGES]
    testLabels = testLabels[0:NUM_TEST_IMAGES]

    return (trainImages, trainLabels), (testImages, testLabels)

def buildModel(images, labels):
    inputSize = len(images[0])

    model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Dense(inputSize, input_shape=(inputSize,)),
        tensorflow.keras.layers.Dense(HIDDEN_LAYER_SIZE, activation = HIDDEN_ACTIVATION),
        tensorflow.keras.layers.Dense(len(LABELS), activation = OUTPUT_ACTIVATION),
    ])

    model.compile(
        optimizer = 'adam',
        loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = ['accuracy']
    )

    return model

def test(model, images, labels):
    loss, accuracy = model.evaluate(images, labels)
    return (loss, accuracy)

def writeFile(path, data):
    with open(path, 'w') as file:
        for row in data:
            file.write('\t'.join([str(item) for item in row]) + "\n")

def writeData(model, trainImages, trainLabels, testImages, testLabels):
    allImages = numpy.concatenate((trainImages, testImages))
    allLabels = numpy.concatenate((trainLabels, testLabels))

    features = []
    neuralTargets = []
    observations = []
    targets = []
    truths = []

    for i in range(len(allImages)):
        features.append([i] + list(allImages[i]))

        for j in range(len(LABELS)):
            neuralTargets.append([i, LABELS[j]])

        if (i < len(trainImages)):
            # Train
            for j in range(len(LABELS)):
                observations.append([i, LABELS[j], int(allLabels[i] == j)])
        else:
            # Test
            for j in range(len(LABELS)):
                targets.append([i, LABELS[j]])

                if (allLabels[i] == j):
                    truths.append([i, LABELS[j]])

    writeFile(FEATURES_PATH, features)
    writeFile(NEURAL_CLASSIFIER_TARGETS_PATH, neuralTargets)

    writeFile(PREDICTED_NUMBER_OBS_PATH, observations)
    writeFile(PREDICTED_NUMBER_TARGETS_PATH, targets)
    writeFile(PREDICTED_NUMBER_TRUTH_PATH, truths)

    writeFile(LABELS_PATH, LABELS)

def main():
    (trainImages, trainLabels), (testImages, testLabels) = loadData()

    os.makedirs(DATA_DIR, exist_ok = True)

    model = buildModel(trainImages, trainLabels)
    model.save(UNTRAINED_MODEL_PATH)

    loss, accuracy = test(model, testImages, testLabels)
    print("Untrained Model -- Loss: %f, Accuracy: %f" % (loss, accuracy))

    model.fit(trainImages, trainLabels, epochs = EPOCHS)
    model.save(TRAINED_MODEL_PATH)

    loss, accuracy = test(model, testImages, testLabels)
    print("Trained Model -- Loss: %f, Accuracy: %f" % (loss, accuracy))

    writeData(model, trainImages, trainLabels, testImages, testLabels)

    pretrainedModel = tensorflow.keras.models.load_model(TRAINED_MODEL_PATH)

    loss, accuracy = test(pretrainedModel, testImages, testLabels)
    print("Pretrained Model -- Loss: %f, Accuracy: %f" % (loss, accuracy))

if (__name__ == '__main__'):
    main()

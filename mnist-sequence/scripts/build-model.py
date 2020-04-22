#!/usr/bin/env python3

# Build a model that is based on classifying a sequence of MNIST digits.

# TODO(eriq): Build a model that takes in all three images.

import datetime
import json
import random
import os

import numpy
import tensorflow

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

# In this path, include the format string for trainingSamplesPerLabel and sequenceInstancesPerLabel.
# All paths will need to make that substitution.
DATA_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'data', 'mnist-sequence', '{:03d}_{:05d}'))

UNTRAINED_MODEL_PATH = os.path.join(DATA_DIR, 'neuralclassifier_model_untrained.h5')
TRAINED_MODEL_PATH = os.path.join(DATA_DIR, 'neuralclassifier_model_trained.h5')
FEATURES_PATH = os.path.join(DATA_DIR, 'neuralclassifier_features.txt')
LABELS_PATH = os.path.join(DATA_DIR, 'neuralclassifier_labels.txt')
NEURAL_CLASSIFIER_TARGETS_PATH = os.path.join(DATA_DIR, 'neuralclassifier_targets.txt')
PREDICTED_NUMBER_OBS_PATH = os.path.join(DATA_DIR, 'predictednumber_obs.txt')
PREDICTED_NUMBER_TARGETS_PATH = os.path.join(DATA_DIR, 'predictednumber_targets.txt')
PREDICTED_NUMBER_TRUTH_PATH = os.path.join(DATA_DIR, 'predictednumber_truth.txt')
SEQUENCES_PATH = os.path.join(DATA_DIR, 'sequence_obs.txt')
MODSUM_PATH = os.path.join(DATA_DIR, 'modsum_obs.txt')
CONFIG_PATH = os.path.join(DATA_DIR, 'config.json')

# This is also the order of the labels in the output layer.
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

HIDDEN_LAYER_SIZE = 512
HIDDEN_ACTIVATION = 'relu'
OUTPUT_ACTIVATION = 'softmax'

SIGNIFICANT_DIGITS = 4

# We don't care if our pretrained model is very accurate.
EPOCHS = 3

# The total number of images in the unlabeled set.
# The sequences are built from this pool.
NUM_TEST = 5000

# The number of training images per label.
NUM_TRAINING_SAMPLES_PER_LABEL = [2, 5, 10, 20]

# The approximate appearances of a label in all of the sequences.
# Appearing twice in a sequence counts for two.
NUM_SEQUENCE_INSTANCES_PER_LABEL = [10, 100, 1000, 10000]

# We will keep cycling through the seeds if there are not enough.
# None mean make a new seed (which may be determined by the previous seed).
SEEDS = [None]

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
    allImages = numpy.concatenate((trainImages, testImages))
    allLabels = numpy.concatenate((trainLabels, testLabels))

    allImages = normalizeImages(allImages)

    return (allImages, allLabels)

def getTraining(numImages, allLabels, trainingSamplesPerLabel):
    indexes = []
    indexesSet = set()

    # {label: count, ...}
    counts = {label: 0 for label in range(len(LABELS))}

    while (len(indexes) < (len(LABELS) * trainingSamplesPerLabel)):
        index = random.randrange(numImages)
        if (index in indexesSet):
            continue

        label = allLabels[index]
        if (counts[label] >= trainingSamplesPerLabel):
            continue
        counts[label] += 1

        indexesSet.add(index)
        indexes.append(index)

    return (indexes, indexesSet)

def getTest(numImages, trainIndexes):
    indexes = []
    indexesSet = set()

    while (len(indexes) < NUM_TEST):
        index = random.randrange(numImages)
        if (index in indexesSet or index in trainIndexes):
            continue

        indexesSet.add(index)
        indexes.append(index)

    return (indexes, indexesSet)

# TODO(eriq): We are starting with addition, but want to include additional sequences.
# Get all the sequences of numbers.
# A seqeunce will be of the form of a tuple of three indexes (indexing into the passed in lists).
# Returns: [(xIndex, yIndex, zIndex), ...]
def buildSequences(allLabels, testIndexes, sequenceInstancesPerLabel):
    # {label (int): [imageIndex, ...], ...}
    imagesByNumber = {label: [] for label in range(len(LABELS))}

    for testIndex in testIndexes:
        number = int(allLabels[testIndex])
        imagesByNumber[number].append(testIndex)

    # Sample the next number randomly, but weighted inversely by the number of times that label has been sampled.
    labelIndexes = [i for i in range(len(LABELS))]
    samplingCounts = [0 for i in range(len(LABELS))]
    samplingWeights = [1.0 / (count + 1) for count in samplingCounts]

    seenSequences = set()

    sequences = []
    while (len(sequences) < int((sequenceInstancesPerLabel * len(LABELS) / 3))):
        # Sample x and y, compute z, then choose images for each digit.
        x = random.choices(labelIndexes, weights = samplingWeights)[0]
        y = random.choices(labelIndexes, weights = samplingWeights)[0]
        z = (x + y) % len(LABELS)

        xIndex = random.choice(imagesByNumber[x])
        yIndex = random.choice(imagesByNumber[y])
        zIndex = random.choice(imagesByNumber[z])

        if (xIndex == yIndex or xIndex == zIndex or yIndex == zIndex):
            continue

        # Make a unique ID for each set of indicies.
        indexID = ((xIndex + 1) * len(LABELS) ** 0) + ((yIndex + 1) * len(LABELS) ** 2) + ((zIndex + 1) * len(LABELS) ** 4)
        if (indexID in seenSequences):
            continue

        seenSequences.add(indexID)
        sequences.append((xIndex, yIndex, zIndex))

        for label in [x, y, z]:
            samplingCounts[label] += 1

        for i in range(len(samplingWeights)):
            samplingWeights[i] = 1.0 / (samplingCounts[i] + 1)

    return sequences

def fetchImages(allImages, allLabels, indexes):
    images = []
    labels = []

    for index in indexes:
        images.append(allImages[index])
        labels.append(allLabels[index])

    return (numpy.array(images), numpy.array(labels))

def buildModel(inputSize):
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

def writeData(model, allImages, allLabels, trainIndexesSet, testIndexesSet, sequences, trainingSamplesPerLabel, sequenceInstancesPerLabel):
    allIndexesSet = trainIndexesSet | testIndexesSet

    features = []
    neuralTargets = []
    observations = []
    targets = []
    truths = []

    for i in allIndexesSet:
        features.append([i] + list(allImages[i]))

        for j in range(len(LABELS)):
            neuralTargets.append([i, LABELS[j]])

        if (i not in testIndexesSet):
            # Train
            for j in range(len(LABELS)):
                observations.append([i, LABELS[j], int(allLabels[i] == j)])
        else:
            # Test
            for j in range(len(LABELS)):
                targets.append([i, LABELS[j]])

                if (allLabels[i] == j):
                    truths.append([i, LABELS[j]])

    # All the possible sequences.
    modsum = []

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            modsum.append((i, j, (i + j) % len(LABELS)))

    writeFile(FEATURES_PATH.format(trainingSamplesPerLabel, sequenceInstancesPerLabel), features)
    writeFile(NEURAL_CLASSIFIER_TARGETS_PATH.format(trainingSamplesPerLabel, sequenceInstancesPerLabel), neuralTargets)

    writeFile(PREDICTED_NUMBER_OBS_PATH.format(trainingSamplesPerLabel, sequenceInstancesPerLabel), observations)
    writeFile(PREDICTED_NUMBER_TARGETS_PATH.format(trainingSamplesPerLabel, sequenceInstancesPerLabel), targets)
    writeFile(PREDICTED_NUMBER_TRUTH_PATH.format(trainingSamplesPerLabel, sequenceInstancesPerLabel), truths)

    writeFile(SEQUENCES_PATH.format(trainingSamplesPerLabel, sequenceInstancesPerLabel), sequences)
    writeFile(MODSUM_PATH.format(trainingSamplesPerLabel, sequenceInstancesPerLabel), modsum)

    writeFile(LABELS_PATH.format(trainingSamplesPerLabel, sequenceInstancesPerLabel), LABELS)

def buildDataset(seed, trainingSamplesPerLabel, sequenceInstancesPerLabel):
    random.seed(seed)

    (allImages, allLabels) = loadData()

    (trainIndexes, trainIndexesSet) = getTraining(len(allImages), allLabels, trainingSamplesPerLabel)
    (testIndexes, testIndexesSet) = getTest(len(allImages), trainIndexes)

    sequences = buildSequences(allLabels, testIndexes, sequenceInstancesPerLabel)

    (trainImages, trainLabels) = fetchImages(allImages, allLabels, trainIndexes)
    (testImages, testLabels) = fetchImages(allImages, allLabels, testIndexes)

    os.makedirs(DATA_DIR.format(trainingSamplesPerLabel, sequenceInstancesPerLabel), exist_ok = True)

    inputSize = len(allImages[0])
    model = buildModel(inputSize)
    model.save(UNTRAINED_MODEL_PATH.format(trainingSamplesPerLabel, sequenceInstancesPerLabel))

    untrainedLoss, untrainedAccuracy = test(model, testImages, testLabels)
    print("Untrained Model (%d, %d) -- Loss: %f, Accuracy: %f" % (trainingSamplesPerLabel, sequenceInstancesPerLabel, untrainedLoss, untrainedAccuracy))

    model.fit(trainImages, trainLabels, epochs = EPOCHS)
    model.save(TRAINED_MODEL_PATH.format(trainingSamplesPerLabel, sequenceInstancesPerLabel))

    trainedLoss, trainedAccuracy = test(model, testImages, testLabels)
    print("Trained ModeDataset (%d, %d) -- Loss: %f, Accuracy: %f" % (trainingSamplesPerLabel, sequenceInstancesPerLabel, trainedLoss, trainedAccuracy))

    writeData(model, allImages, allLabels, trainIndexesSet, testIndexesSet, sequences, trainingSamplesPerLabel, sequenceInstancesPerLabel)

    pretrainedModel = tensorflow.keras.models.load_model(TRAINED_MODEL_PATH.format(trainingSamplesPerLabel, sequenceInstancesPerLabel))

    trainedLoss, trainedAccuracy = test(pretrainedModel, testImages, testLabels)
    print("Pretrained Model (%d, %d) -- Loss: %f, Accuracy: %f" % (trainingSamplesPerLabel, sequenceInstancesPerLabel, trainedLoss, trainedAccuracy))

    config = {
        'network': {
            'hiddenLayerSize': HIDDEN_LAYER_SIZE,
            'hiddenLayerActivation': HIDDEN_ACTIVATION,
            'outputActivation': OUTPUT_ACTIVATION,
        },
        'untrained': {
            'loss': untrainedLoss,
            'accuracy': untrainedAccuracy,
        },
        'pretrained': {
            'epochs': EPOCHS,
            'loss': trainedLoss,
            'accuracy': trainedAccuracy,
        },
        'testSamples': NUM_TEST,
        'trainingSamplesPerLabel': trainingSamplesPerLabel,
        'sequenceInstanesPerLabel': sequenceInstancesPerLabel,
        'seed': seed,
        'timestamp': str(datetime.datetime.now()),
    }

    with open(CONFIG_PATH.format(trainingSamplesPerLabel, sequenceInstancesPerLabel), 'w') as file:
        json.dump(config, file, indent = 4)

def main():
    random.seed()

    count = 0
    for trainingSamplesPerLabel in NUM_TRAINING_SAMPLES_PER_LABEL:
        for sequenceInstancesPerLabel in NUM_SEQUENCE_INSTANCES_PER_LABEL:
            seed = SEEDS[count % len(SEEDS)]
            if (seed is None):
                seed = random.randrange(2 ** 64)
            count += 1

            buildDataset(seed, trainingSamplesPerLabel, sequenceInstancesPerLabel)

if (__name__ == '__main__'):
    main()

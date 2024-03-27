import matplotlib

matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from keras import layers, optimizers, models, Sequential
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os


ap = argparse.ArgumentParser()

ap.add_argument(
    "-d", "--dataset", required=True, help="Path to input dataset of images"
)
ap.add_argument("-m", "--model", required=True, help="path to output trained model")
ap.add_argument(
    "-l", "--label-bin", required=True, help="path to output label binarizer"
)
ap.add_argument("-p", "--plot", required=True, help="path to output accuracy/loss plot")

args = vars(ap.parse_args())

print("[INFO] loading images...")
data = []
labels = []

imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32)).flatten()
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
label = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.25, random_state=42
)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print(lb.classes_)

model = Sequential()
model.add(layers.Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(layers.Dense(512, activation="sigmoid"))

model.add(layers.Dense(1, activation="sigmoid"))

# We use this one in case we need to clasify more than one thing
# model.add(layers.Dense(len(lb.classes_), activation="softmax"))

INIT_LR = 0.01
EPOCHS = 80

print("[INFO] training network...")
opt = optimizers.SGD(learning_rate=INIT_LR)

# in case we need to classify more than one thing we swap binary_crossentropy with categorical_crossentropy
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(
    x=trainX, y=trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32
)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=32)
binary_predictions = np.round(predictions)

print(
    classification_report(
        testY.argmax(axis=1),
        binary_predictions,
        target_names=lb.classes_,
        zero_division=1,
    )
)
print("crap")
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

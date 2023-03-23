# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:54:06 2022

@author: damjan.janakievski
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image, ImageOps, ImageFilter
# initialize the initial learning rate, number of epochs to train for and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r"C:\\Users\\damjan.janakievski\\OneDrive - A1 Group\\Desktop\\Face-Mask-Detector2-master\\dataset"
CATEGORIES = ["with_mask", "without_mask"]
# grab the list of images in our dataset directory, then initialize
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = Image.open(img_path)
        # gray_image = ImageOps.grayscale(image)
        print(image.mode)
        resizedImage = image.resize((224, 224))
        print(resizedImage.mode)
        back_to_color_image = resizedImage.convert("RGB")
        print(back_to_color_image.mode)
        noisy_image = back_to_color_image.filter(ImageFilter.GaussianBlur(4))
        print(noisy_image.mode)
        
        
        # matrix = (0.2, 0.5, 0.3, 0.0, 0.2, 0.5, 0.3, 0.0, 0.2, 0.5, 0.3, 0.0)
        # back_to_color_image_1 = back_to_color_image_2.convert('RGB',matrix)
        
        # image.show()
        # gray_image.show()
        # back_to_color_image_1.show()
        # back_to_color_image_2.show()
        print('kraj')
        image = img_to_array(noisy_image)
        image = preprocess_input(image)
        
        # adding a noise

        
        
        data.append(image)
        labels.append(category)
        # print('The size of thumbnail image is: ', image.size)
        # import time
        # time.sleep(5)




# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")



# load the MobileNetV2 network, ensuring the head FC layer sets are left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

# head of the model 
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)


model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# train the head of the network\
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# make predictions on the testing set
predIdxs = model.predict(testX, batch_size=BS)

predIdxs = np.argmax(predIdxs, axis=1)

# classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
    target_names=lb.classes_))

model.save("mask_detector_122x122_gray_noisy.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

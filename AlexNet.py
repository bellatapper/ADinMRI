import tensorflow as tf
from tensorflow.keras import datasets, models, layers
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
import nibabel as nib 
import os
from sklearn.utils import shuffle
import glob

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# for filename in os.listdir("./AD"):
#     if filename.endswith(".mgz"):
#         print(filename)
#         nii_obj = nib.load("./AD/" + filename)
#         images = nii_obj.get_fdata()[28:228,110:130,28:228].transpose(1,2,0)
#         images = images.astype(np.uint8)
#         for i in range(20):
#             image = np.expand_dims(images, axis=3)
#             image = np.repeat(image, 3,axis=3)
#             img = Image.fromarray(image[i], 'RGB' )
#             img.save("./Images/AD/" + filename + str(i) + ".png")
#     else:
#         continue

# for filename in os.listdir("./CN"):
#     if filename.endswith(".mgz"):
#         print(filename)
#         nii_obj = nib.load("./CN/" + filename)
#         images = nii_obj.get_fdata()[28:228,110:130,28:228].transpose(1,2,0)
#         images = images.astype(np.uint8)
#         for i in range(20):
#             image = np.expand_dims(images, axis=3)
#             image = np.repeat(image, 3,axis=3)
#             img = Image.fromarray(image[i], 'RGB' )
#             img.save("./Images/CN/" + filename + str(i) + ".png")
#     else:
#         continue


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
        samplewise_center=True,
        samplewise_std_normalization=True)

train_generator = train_datagen.flow_from_directory(
        './Images',
        target_size=(200, 200),
        batch_size=32,
        class_mode='binary',
        shuffle=True,
        seed=42,
        subset='training')

validation_generator = train_datagen.flow_from_directory(
    './Images', # same directory as training data
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary',
    subset='validation') # set as validation data



modelInput = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(200,200,3))

x = modelInput.output
x = layers.Flatten()(x)
x = layers.Dense(units=256,activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(units=256,activation="relu")(x)
x = layers.Dropout(0.5)(x)
modelOut = layers.Dense(units=1, activation="sigmoid")(x)

model = tf.keras.models.Model(inputs=modelInput.input, outputs=modelOut)

print(model.summary())

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.9)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.fit_generator(generator=train_generator,validation_data = validation_generator, 
                    epochs=100
)

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
import sklearn as skl
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D,
                                     Flatten, Dropout, DepthwiseConv2D, SeparableConv2D,
                                     Activation, BatchNormalization, SpatialDropout2D,
                                     concatenate)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical

tfd = tfp.distributions
tfpl = tfp.layers

# Load MNIST data
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Use the EMNIST digit datasets
dspath = '/mnt/g/WSL/downloaded_ml_data/emnist-all/'
train, test = (
    pd.read_csv(dspath+'emnist-digit-train.csv'),
    pd.read_csv(dspath+"emnist-digit-test.csv")
    )

train_labels, test_labels = train.iloc[:, 0].values, test.iloc[:, 0].values
train_images, test_images = (
    train.iloc[:, 1:].values.reshape(train.shape[0], 28, 28, 1),
    test.iloc[:, 1:].values.reshape(test.shape[0], 28, 28, 1)
    )

# Normalize pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0
print(train_images.shape, train_labels.shape)

# Reshape images to have single-channel
# train_images, test_images = (train_images.reshape(train_images.shape[0], 28, 28, 1),
#                              test_images.reshape(test_images.shape[0], 28, 28, 1))
# One-hot encode labels
train_labels, test_labels = to_categorical(train_labels), to_categorical(test_labels)

def neg_loglike(ytrue, ypred):
    return -ypred.log_prob(ytrue)

def divergence(q,p,_):
    return tfd.kl_divergence(q,p)/train_labels.shape[0]

def create_bnn():
    model = Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical', input_shape=(28, 28, 1)),
    # tf.keras.layers.RandomRotation(0.15, input_shape=(28, 28, 1)),
    tf.keras.layers.RandomTranslation(0.2, 0.2, input_shape=(28, 28, 1)),
    # tf.keras.layers.RandomContrast(0.1, input_shape=(28, 28, 1)),
    # tf.keras.layers.RandomBrightness(0.1),
    # Flatten(),
    tfpl.Convolution2DFlipout(32,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        kernel_divergence_fn=divergence,
        bias_divergence_fn=divergence, ),

    MaxPooling2D((2, 2)),
    Dropout(0.1),
    tfpl.Convolution2DFlipout(48,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_divergence_fn=divergence,
        bias_divergence_fn=divergence, ),

    Dropout(0.1),
    MaxPooling2D((2, 2)),
    tfpl.Convolution2DFlipout(64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        kernel_divergence_fn=divergence,
        bias_divergence_fn=divergence, ),

    Dropout(0.1),
    # MaxPooling2D((2,2)),
    Flatten(),
    tfpl.DenseFlipout(512,
                      activation='relu',
                      kernel_divergence_fn=divergence,
                      bias_divergence_fn=divergence),
    Dropout(0.1),
    tfpl.DenseFlipout(64,
                      activation='relu',
                      kernel_divergence_fn=divergence,
                      bias_divergence_fn=divergence),
    Dropout(0.1),
    tfpl.DenseFlipout(10,
                      activation='relu',
                      kernel_divergence_fn=divergence,
                      bias_divergence_fn=divergence),

    tfpl.OneHotCategorical(10,convert_to_tensor_fn=tfd.Distribution.mode)
    ])
    return model

model = create_bnn()
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=neg_loglike,
              metrics=['accuracy'],
              experimental_run_tf_function=False
              )

print(model.summary())

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    start_from_epoch=25,
    restore_best_weights=True,
    mode='max'
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor = 0.95,
    patience=5,
    cooldown=5,
    mode='max',
    min_lr=1e-7,
    verbose=1,
    min_delta=0.01)

class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(np.argmax(train_labels,axis=1)),
                                                  y=np.argmax(train_labels,axis=1))
cws = {}
for num in range(10):
    cws[num] = class_weights[num]
mdlhist = model.fit(train_images,
                    train_labels,
                    # class_weight=cws,
                    batch_size=1024,
                    epochs=200,
                    validation_data=(test_images, test_labels),
                    callbacks=[earlystop, reduce_lr])

print("Evaluating with EMNIST test set")
model.evaluate(test_images, test_labels)
print(classification_report(test_labels, model.predict(test_images)))
#
# print("#"*40)
# print("Evaluating with Orig MNIST test set")
#
# # Load MNIST data
# # Use the EMNIST digit datasets
# dspath = '/mnt/g/WSL/downloaded_ml_data/emnist-all/'
# mtrain, mtest = (
#     pd.read_csv(dspath+'emnist-mnist-train.csv'),
#     pd.read_csv(dspath+"emnist-mnist-test.csv")
#     )
#
# mtrain_labels, mtest_labels = mtrain.iloc[:, 0].values, mtest.iloc[:, 0].values
# mtrain_images, mtest_images = (
#     mtrain.iloc[:, 1:].values.reshape(mtrain.shape[0], 28, 28, 1),
#     mtest.iloc[:, 1:].values.reshape(mtest.shape[0], 28, 28, 1)
#     )
#
# # Normalize pixel values
# mtrain_images, mtest_images = mtrain_images / 255.0, mtest_images / 255.0
# print(mtrain_images.shape, mtrain_labels.shape)
#
# # Reshape images to have single-channel
# # train_images, test_images = (train_images.reshape(train_images.shape[0], 28, 28, 1),
# #                              test_images.reshape(test_images.shape[0], 28, 28, 1))
# # One-hot encode labels
# mtrain_labels, mtest_labels = to_categorical(mtrain_labels), to_categorical(mtest_labels)
# model.evaluate(mtest_images, mtest_labels)
# print(classification_report(mtest_labels, model.predict(mtest_images)))

model.save('/home/lreclusa/repositories/BNN-OCR-WebApp/mnist_bnn')
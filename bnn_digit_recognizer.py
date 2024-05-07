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
    pd.read_csv(dspath+'emnist-balanced-train.csv'),
    pd.read_csv(dspath+"emnist-balanced-test.csv")
    )

train_labels, test_labels = train.iloc[:, 0].values, test.iloc[:, 0].values
train_images, test_images = (
    train.iloc[:, 1:].values.reshape(-1, 28, 28, 1),
    test.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    )


bal_maps = pd.read_csv(dspath+'emnist-balanced-mapping.csv')
# Normalize pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0
print(train_images.shape, train_labels.shape)

# Reshape images to have single-channel
# train_images, test_images = (train_images.reshape(train_images.shape[0], 28, 28, 1),
#                              test_images.reshape(test_images.shape[0], 28, 28, 1))
# One-hot encode labels
n_labels = len(np.unique(test_labels))
lbl_names = [chr(int(bal_maps.values[ii, 1])) for ii in np.unique(test_labels)]
print(lbl_names)
train_labels, test_labels = to_categorical(train_labels), to_categorical(test_labels)

def neg_loglike(ytrue, ypred):
    return -ypred.log_prob(ytrue)

def divergence(q,p,_):
    return tfd.kl_divergence(q,p)/train_labels.shape[0]

def create_bnn():
    model = Sequential([
    tf.keras.layers.RandomFlip('horizontal', input_shape=(28, 28, 1)),
    # tf.keras.layers.RandomRotation(0.05, input_shape=(28, 28, 1)),
    # tf.keras.layers.RandomTranslation(0.1, 0.1, input_shape=(28, 28, 1)),
    # tf.keras.layers.RandomContrast(0.1, input_shape=(28, 28, 1)),
    # tf.keras.layers.RandomBrightness(0.1),
    # Flatten(),
    tfpl.Convolution2DFlipout(128,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        activation='relu',
        kernel_divergence_fn=divergence,
        bias_divergence_fn=divergence, ),
    # SpatialDropout2D(0.05),

    MaxPooling2D((2, 2), padding='same'),

    tfpl.Convolution2DFlipout(64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu',
        kernel_divergence_fn=divergence,
        bias_divergence_fn=divergence, ),
    # SpatialDropout2D(0.01),


    MaxPooling2D((2, 2), padding='same'),

    tfpl.Convolution2DFlipout(32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu',
        kernel_divergence_fn=divergence,
        bias_divergence_fn=divergence, ),
    # SpatialDropout2D(0.01),

    # MaxPooling2D((2, 2), padding='same'),

    MaxPooling2D((2, 2), padding='same'),

    Flatten(),
    tfpl.DenseFlipout(512,
                      activation='relu',
                      kernel_divergence_fn=divergence,
                      bias_divergence_fn=divergence),
    Dropout(0.5),
    tfpl.DenseFlipout(128,
                      activation='relu',
                      kernel_divergence_fn=divergence,
                      bias_divergence_fn=divergence),
    Dropout(0.5),
    tfpl.DenseFlipout(n_labels,
                      activation='relu',
                      kernel_divergence_fn=divergence,
                      bias_divergence_fn=divergence),

    tfpl.OneHotCategorical(n_labels,convert_to_tensor_fn=tfd.Distribution.mode)
    ])
    return model



model = create_bnn()
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss=neg_loglike,
              metrics=['accuracy'],
              experimental_run_tf_function=False
              )

print(model.summary())

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
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
                                                  classes=np.unique(np.argmax(train_labels, axis=1)),
                                                  y=np.argmax(train_labels, axis=1))
cws = {}
for num in range(10):
    cws[num] = class_weights[num]
mdlhist = model.fit(train_images,
                    train_labels,
                    # class_weight=cws,
                    batch_size=512,
                    epochs=200,
                    validation_data=(test_images, test_labels),
                    callbacks=[reduce_lr, earlystop])

print("Evaluating with EMNIST test set")
model.evaluate(test_images, test_labels)
mdl_preds = np.argmax(model.predict(test_images), axis=1)

print(classification_report(
    np.argmax(test_labels, axis=1),
    mdl_preds,
    target_names=lbl_names
    )
)

# cm = skl.metrics.confusion_matrix(np.argmax(test_labels, axis=1), mdl_preds)
from matplotlib import rcParams

rcParams['font.size'] = 8
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

fig,ax = plt.subplots(figsize=(10,10))
cm_plot = skl.metrics.ConfusionMatrixDisplay.from_predictions(
    np.argmax(test_labels, axis=1),
    mdl_preds,
    display_labels=lbl_names,
    cmap='Blues',
    ax=ax,
    colorbar=False
)
plt.savefig("/mnt/g/WSL/wsl_ml_figures/emnist-balanced-cm-plot.png",dpi=150, bbox_inches='tight')
plt.close()
print("#"*40)

# Load MNIST data
# Use the EMNIST digit datasets
dspath = '/mnt/g/WSL/downloaded_ml_data/emnist-all/'
mtrain, mtest = (
    pd.read_csv(dspath+'emnist-mnist-train.csv'),
    pd.read_csv(dspath+"emnist-mnist-test.csv")
    )

mtrain_labels, mtest_labels = mtrain.iloc[:, 0].values, mtest.iloc[:, 0].values
mtrain_images, mtest_images = (
    mtrain.iloc[:, 1:].values.reshape(-1, 28, 28, 1),
    mtest.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    )

# Normalize pixel values
mtrain_images, mtest_images = mtrain_images / 255.0, mtest_images / 255.0
print(mtrain_images.shape, mtrain_labels.shape)

if n_labels == 10:
    print("Evaluating with Orig MNIST test set")

    # Reshape images to have single-channel
    train_images, test_images = (train_images.reshape(train_images.shape[0], 28, 28, 1),
                                 test_images.reshape(test_images.shape[0], 28, 28, 1))
    # One-hot encode labels
    mtrain_labels, mtest_labels = to_categorical(mtrain_labels), to_categorical(mtest_labels)
    model.evaluate(mtest_images, mtest_labels)
    print(classification_report(mtest_labels, model.predict(mtest_images)))

model.save('/home/lreclusa/repositories/BNN-OCR-WebApp/mnist_bnn')
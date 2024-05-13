import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
import sklearn as skl
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import pickle as pk

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
from tensorflow.keras import backend as K

tfd = tfp.distributions
tfpl = tfp.layers

# Load MNIST data
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def setup_training_env():
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

    def neg_loglike(ytrue, ypred):
        return -ypred.log_prob(ytrue)

    def divergence(q, p, _):
        return tfd.kl_divergence(q, p) / train_labels.shape[0]

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

    train_datagen = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(1024)
    test_datagen = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(10000).batch(1024)

    print(train_images.shape, test_images.shape)



    return (
        train_datagen,
        test_datagen,
        {
            'train_images': train_images,
            'test_images':test_images,
            'train_labels': train_labels,
            'test_labels':test_labels,
            'lbl_names': lbl_names,
            'n_labels': n_labels,
            'bal_maps': bal_maps,
            'neg_loglike': neg_loglike,
            'divergence': divergence
        }
    )



def neg_loglike(ytrue, ypred):
    return -ypred.log_prob(ytrue)

def divergence(q, p, _):
    return tfd.kl_divergence(q, p) / 112799.


def create_bnn(n_labels):
    # global n_labels

    model = Sequential([
    tf.keras.layers.RandomFlip('horizontal', input_shape=(28, 28, 1)),
    tf.keras.layers.RandomRotation(0.1, input_shape=(28, 28, 1)),
    # tf.keras.layers.RandomTranslation(0.1, 0.1, input_shape=(28, 28, 1)),
    # tf.keras.layers.RandomContrast(0.1, input_shape=(28, 28, 1)),
    # tf.keras.layers.RandomBrightness(0.1),
    # Flatten(),

    tfpl.Convolution2DFlipout(48,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='linear',
        kernel_divergence_fn=divergence,
        bias_divergence_fn=divergence, ),
    # SpatialDropout2D(0.05),
    Activation("relu"),


    MaxPooling2D((2, 2), padding='same'),

    Conv2D(64,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        activation='linear'),
    # # SpatialDropout2D(0.01),
    #
    Activation("relu"),

    MaxPooling2D((2, 2), padding='same'),

    Conv2D(128,
        kernel_size=(7, 7),
        strides=(1, 1),
        padding='same',
        activation='linear'),
    # # SpatialDropout2D(0.01),
    #
    Activation("relu"),
    # MaxPooling2D((2, 2), padding='same'),

    Flatten(),
    Dense(1024, activation='linear'),
    # Dropout(0.05),
    # tfpl.DenseFlipout(64,
    #                   activation='linear',
    #                   kernel_divergence_fn=divergence,
    #                   bias_divergence_fn=divergence),
    Activation("relu"),
    Dropout(0.5),


    # tfpl.DenseFlipout(64,
    #                   activation='linear',
    #                   kernel_divergence_fn=divergence,
    #                   bias_divergence_fn=divergence),
    # # Dropout(0.05),
    # # tfpl.DenseFlipout(64,
    # #                   activation='linear',
    # #                   kernel_divergence_fn=divergence,
    # #                   bias_divergence_fn=divergence),
    # Activation("relu"),
    # Dropout(0.1),

    tfpl.DenseFlipout(n_labels,
                      activation='relu',
                      kernel_divergence_fn=divergence,
                      bias_divergence_fn=divergence),

    tfpl.OneHotCategorical(n_labels,convert_to_tensor_fn=tfd.Distribution.mode)
    ])
    return model



    # class_weights = class_weight.compute_class_weight(class_weight='balanced',
    #                                                   classes=np.unique(np.argmax(train_labels, axis=1)),
    #                                                   y=np.argmax(train_labels, axis=1))

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    start_from_epoch=50,
    restore_best_weights=True,
    mode='max'
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=10,
    cooldown=5,
    mode='max',
    min_lr=1e-7,
    verbose=1,
    min_delta=0.01)





def main():
    train_datagen, test_datagen, env_dict = setup_training_env()
    n_labels = env_dict['n_labels']
    lbl_names = env_dict['lbl_names']
    divergence  = env_dict['divergence']
    neg_loglike = env_dict['neg_loglike']

    model = create_bnn(n_labels)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=neg_loglike,
                  metrics=['accuracy'],
                  experimental_run_tf_function=False
                  )

    print(model.summary())
    mdlhist = model.fit(train_datagen,
                        # class_weight=cws,
                        # batch_size=256,
                        epochs=200,
                        validation_data=test_datagen,
                        callbacks=[reduce_lr, earlystop])

    test_images = env_dict['test_images']
    test_labels = env_dict['test_labels']

    print("Evaluating with EMNIST test set")
    model.evaluate(test_images, test_labels)
    mdl_preds = np.argmax(model.predict(test_images), axis=1)

    print(classification_report(
        np.argmax(test_labels, axis=1),
        mdl_preds,
        target_names=env_dict['lbl_names']
        )
    )
    mdl_weights = model.get_weights()

    with open("ocr_bnn_weights.pk", "wb") as f:
        pk.dump(mdl_weights, f)

    model.save('/home/lreclusa/repositories/BNN_OCR-WebApp/mnist_bnn')
    model.save_weights(
        '/home/lreclusa/repositories/BNN-OCR-WebApp/mnist_bnn/mnist_bnn_weights.weights.h5',
        overwrite=False)

    # cm = skl.metrics.confusion_matrix(np.argmax(test_labels, axis=1), mdl_preds)
    from matplotlib import rcParams

    rcParams['font.size'] = 8
    rcParams['axes.labelsize'] = 14
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14

    fig, ax = plt.subplots(figsize=(10, 10))
    cm_plot = skl.metrics.ConfusionMatrixDisplay.from_predictions(
        np.argmax(test_labels, axis=1),
        mdl_preds,
        display_labels=lbl_names,
        cmap='Blues',
        ax=ax,
        colorbar=False
    )
    plt.savefig("/mnt/g/WSL/wsl_ml_figures/emnist-balanced-cm-plot.png", dpi=150, bbox_inches='tight')
    plt.close()

    return (model, env_dict, mdl_preds)
# print("#"*40)
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
#     mtrain.iloc[:, 1:].values.reshape(-1, 28, 28, 1),
#     mtest.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
#     )
#
# # Normalize pixel values
# mtrain_images, mtest_images = mtrain_images / 255.0, mtest_images / 255.0
# print(mtrain_images.shape, mtrain_labels.shape)
#
# if n_labels == 10:
#     print("Evaluating with Orig MNIST test set")
#
#     # Reshape images to have single-channel
#     train_images, test_images = (train_images.reshape(train_images.shape[0], 28, 28, 1),
#                                  test_images.reshape(test_images.shape[0], 28, 28, 1))
#     # One-hot encode labels
#     mtrain_labels, mtest_labels = to_categorical(mtrain_labels), to_categorical(mtest_labels)
#     model.evaluate(mtest_images, mtest_labels)
#     print(classification_report(mtest_labels, model.predict(mtest_images)))


if __name__ == "__main__":
    model, env_dict, mdl_preds = main()
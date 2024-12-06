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

def setup_training_env(bs=2200):
    # Use the EMNIST digit datasets
    dataset_type='bymerge'
    dspath = '/mnt/g/WSL/downloaded_ml_data/emnist-all/'
    train, test = (
        pd.read_csv(dspath+f'emnist-{dataset_type}-train.csv'),
        pd.read_csv(dspath+f"emnist-{dataset_type}-test.csv")
        )
    train, test = train.sample(frac=0.6, random_state=999, axis=0), test.sample(frac=1, random_state=999, axis=0)

    train_labels, test_labels = (
        train.iloc[:, 0].values,
        test.iloc[:, 0].values
    )
    train_images, test_images = (
        train.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype('uint8'),
        test.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype('uint8')
    )

    def neg_loglike(ytrue, ypred):
        return -ypred.log_prob(ytrue)

    def divergence(q, p, _):
        return tfd.kl_divergence(q, p) / train_labels.shape[0]

    bal_maps = pd.read_csv(dspath+f'emnist-balanced-mapping.csv')
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

    train_datagen = tf.data.Dataset.from_tensor_slices((train_images.astype('float16'), train_labels)).shuffle(300000).batch(bs)
    test_datagen = tf.data.Dataset.from_tensor_slices((test_images.astype('float16'), test_labels)).shuffle(100000).batch(bs)

    print(train_images.shape, test_images.shape)

    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(np.argmax(train_labels, axis=1)),
                                                      y=np.argmax(train_labels, axis=1))
    class_weights = {int(i): cw for i, cw in enumerate(class_weights)}
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
            'divergence': divergence,
            'class_weights': class_weights
        }
    )



# def neg_loglike(ytrue, ypred):
#     return -ypred.log_prob(ytrue)
#
# def divergence(q, p, _):
#     return tfd.kl_divergence(q, p) / 112799.


def create_bnn(n_labels=47, DO=0.25, divergence=None):
    # global n_labels

    model = Sequential([
    tf.keras.layers.RandomFlip('horizontal', input_shape=(28, 28, 1)),
    # tf.keras.layers.RandomRotation(0.05, input_shape=(28, 28, 1)),
    tf.keras.layers.RandomTranslation(0.15, 0.15, input_shape=(28, 28, 1)),
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

    # tfpl.Convolution2DFlipout(48,
    #     kernel_size=(7, 7),
    #     strides=(1, 1),
    #     padding='same',
    #     activation='linear',
    #     kernel_divergence_fn=divergence,
    #     bias_divergence_fn=divergence, ),
    # # SpatialDropout2D(0.05),
    # Activation("relu"),
    # SpatialDropout2D(0.1),

    Conv2D(48,
        kernel_size=(1, 3),
        strides=(1, 1),
        padding='same',
        activation='linear'),
    Activation("relu"),

    Conv2D(48,
        kernel_size=(3, 1),
        strides=(1, 1),
        padding='same',
        activation='linear'),
    Activation("relu"),

    MaxPooling2D((2, 2), padding='same'),

    Conv2D(64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='linear'),
    Activation("relu"),
    # SpatialDropout2D(0.25),

    Conv2D(64,
        kernel_size=(1, 3),
        strides=(1, 1),
        padding='same',
        activation='linear'),
    Activation("relu"),
    # SpatialDropout2D(0.25),

    Conv2D(64,
        kernel_size=(3, 1),
        strides=(1, 1),
        padding='same',
        activation='linear'),
    Activation("relu"),
    # SpatialDropout2D(0.25),
    MaxPooling2D((2, 2), padding='same'),

    Conv2D(128,
        kernel_size=(1, 3),
        strides=(1, 1),
        padding='same',
        activation='linear'),
    Activation("relu"),

    Conv2D(128,
        kernel_size=(3, 1),
        strides=(1, 1),
        padding='same',
        activation='linear'),
    Activation("relu"),

    Conv2D(128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='linear'),

    Activation("relu"),
    SpatialDropout2D(0.25),

    MaxPooling2D((2, 2), padding='same'),

    # Conv2D(256,
    #     kernel_size=(3, 3),
    #     strides=(1, 1),
    #     padding='same',
    #     activation='linear'),
    # Activation("relu"),

    # BatchNormalization(),
    Flatten(),
    Dense(512, activation='linear'),
    # Dropout(0.05),
    # tfpl.DenseFlipout(64,
    #                   activation='linear',
    #                   kernel_divergence_fn=divergence,
    #                   bias_divergence_fn=divergence),
    Activation("relu"),
    Dropout(DO),

    Dense(64, activation='linear'),
    # Dropout(0.05),
    # tfpl.DenseFlipout(64,
    #                   activation='linear',
    #                   kernel_divergence_fn=divergence,
    #                   bias_divergence_fn=divergence),
    Activation("relu"),
    Dropout(DO),
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

    tfpl.OneHotCategorical(n_labels, convert_to_tensor_fn=tfd.Distribution.mode)
    ])
    return model






earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    start_from_epoch=150,
    restore_best_weights=True,
    mode='max'
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.75,
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
    class_weights = env_dict['class_weights']

    model = create_bnn(**{'n_labels': n_labels,
                          'divergence': divergence,
                          'DO': 0.2})
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=5e-4,
            beta_1=0.95,
            beta_2=0.9,
            weight_decay=7.5e-2,
        ),
        loss=neg_loglike,
        metrics=['accuracy'],
        experimental_run_tf_function=False
                  )

    print(model.summary())
    mdlhist = model.fit(train_datagen.repeat(),
                        class_weight=class_weights,
                        # batch_size=256,
                        epochs=300,
                        steps_per_epoch=400,
                        validation_data=test_datagen,
                        callbacks=[reduce_lr, earlystop])

    test_images = env_dict['test_images']
    test_labels = env_dict['test_labels']

    print("Evaluating with EMNIST test set")
    # model.evaluate(test_images, test_labels)
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
    # model.save_weights(
    #     '/home/lreclusa/repositories/BNN-OCR-WebApp/mnist_bnn/mnist_bnn_weights.weights.h5',
    #     overwrite=False)

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
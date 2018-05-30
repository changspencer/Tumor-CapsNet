import h5py
import glob
import random
import numpy as np
from keras import models, layers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from capsnetKeras.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask


def get_train_data():
    train_data, train_labels = [], []
    folders = ['RawData/brainTumorDataPublic_1766',
               'RawData/brainTumorDataPublic_7671532',
               'RawData/brainTumorDataPublic_15332298',
               'RawData/brainTumorDataPublic_22993064']
    for fold in folders:
        list_files = glob.glob(fold + "/*.mat")
        random.shuffle(list_files)
        print("Getting from files in {}...".format(fold))
        for file in list_files:
            with h5py.File(file) as f:
                # print("Getting segmented image and label from {}"
                #       .format(file))
                img = f["cjdata/image"]
                mask = f["cjdata/tumorMask"]
                train_labels.append(int(f["cjdata/label"][0]))
                img = np.array(img)
                mask = np.array(mask)
                # Normalize to 0.0 to 1.0
                img = img * (1 / np.max(np.max(img)))
                train_data.append(np.multiply(img, mask))
    train_data = np.asarray(train_data)
    train_labels = np.asarray(train_labels)
    train_labels = to_categorical(train_labels, num_classes=4)
    train_labels = train_labels[:, 1:]

    return train_data, train_labels


def build_model():
    number_of_classes = 3
    input_shape = (64, 64, 1)

    x = layers.Input(shape=input_shape)
    '''
    Inputs to the model are MRI images which are down-sampled
    to 64 × 64 from 512 × 512, in order to reduce the number of
    parameters in the model and decrease the training time.
    Second (First?) layer is a convolutional layer with 64 × 9 × 9 filters
    and stride of 1 which leads to 64 feature maps of size 56×56.
    '''
    conv1 = layers.Conv2D(64, (9, 9), activation='relu',
                          name="FirstLayer")(x)
    '''
    The second layer is a Primary Capsule layer resulting from
    256×9×9 convolutions with strides of 2.
    '''
    primaryCaps = PrimaryCap(inputs=conv1, dim_capsule=8,
                             n_channels=32, kernel_size=9, strides=2,
                             padding='valid')
    '''
    This layer consists of 32 “Component Capsules” with dimension of 8 each of
    which has feature maps of size 24×24 (i.e., each Component
    Capsule contains 24 × 24 localized individual Capsules).
    '''
    # capLayer1 = CapsuleLayer(
    # num_capsule=32, dim_capsule=8, routings=3, name="SecondLayer")(primaryCaps)
    # num_capsule=4, dim_capsule=8, routings=3, name="SecondLayer")(primaryCaps)
    '''
    Final capsule layer includes 3 capsules, referred to as “Class
    Capsules,’ ’one for each type of candidate brain tumor. The
    dimension of these capsules is 16.
    '''
    capLayer2 = CapsuleLayer(num_capsule=3, dim_capsule=16, routings=2,
                             name="ThirdLayer")(primaryCaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its
    # length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(capLayer2)

    # Decoder network.
    y = layers.Input(shape=(number_of_classes,))
    # The true label is used to mask the output of capsule layer. For training
    masked_by_y = Mask()([capLayer2, y])

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu',
                             input_dim=16 * number_of_classes))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])

    train_model.compile(optimizer="rmsprop", loss='mse', metrics=['accuracy'])

    return train_model


def k_fold_validation(model, train_data, train_labels, num_folds):
    checkpointer = ModelCheckpoint(filepath='CapsNet.h5',
                                   monitor='val_acc',
                                   save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_acc', patience=3)

    mse_results = np.zeros(num_folds)
    for fold in range(num_folds):
        hst = model.fit(train_data,
                        train_labels,
                        epochs=20,
                        verbose=1,
                        callbacks=[early_stopping,
                                   checkpointer],
                        validation_split=0.25)
        mse_results[fold] = hst.history['val_mse']

    return mse_results, np.mean(mse_results)


def main():
    train_data, train_labels = get_train_data()
    model = build_model()
    num_folds = 5
    results_per_fold, mean = k_fold_validation(model, train_data,
                                               train_labels, num_folds)
    print("Results per fold:", results_per_fold)
    print("Mean of results:", mean)


if __name__ == '__main__':
    main()

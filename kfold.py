import h5py
import glob
import random
import json
import matplotlib
import numpy as np
from skimage import transform
from keras import models, layers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from capsnetKeras.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def prepare_data():
    train_data, train_labels = [], []
    list_files = []
    folders = ['RawData/brainTumorDataPublic_1766',
               'RawData/brainTumorDataPublic_7671532',
               'RawData/brainTumorDataPublic_15332298',
               'RawData/brainTumorDataPublic_22993064']
    print("Preparing the brain tumor data...")
    for fold in folders:
        list_files += glob.glob(fold + "/*.mat")
    random.shuffle(list_files)
    for file in list_files:
        with h5py.File(file) as f:
            img = f["cjdata/image"]
            mask = f["cjdata/tumorMask"]
            # Labels are 1-indexed
            train_labels.append(int(f["cjdata/label"][0]) - 1)
            img = np.array(img)
            mask = np.array(mask)
            # Normalize to 0.0 to 1.0
            img = img * (1 / np.max(np.max(img)))
            seg_img = np.multiply(img, mask)
            # seg_img = img
            seg_img = transform.resize(seg_img, (64, 64))
            train_data.append(seg_img)
    train_data = np.asarray(train_data)
    print("train_data.shape = ", train_data.shape)
    train_labels = np.asarray(train_labels)
    train_labels = to_categorical(train_labels, num_classes=3)
    print("Sample Train_labels: {}".format(train_labels[0]))
    # train_labels = train_labels[:, 1:]
    # print("Sample Train_labels (mod): {}".format(train_labels[0]))
    print("train_labels.shape = ", train_labels.shape)
    print("Done preparing data")
    return train_data, train_labels

# def get_train_data():
#     data_list = []
#     with open(data_file) as f:
#         data_list = [json.loads(line) for line in f]
#     np.asarray(data_list)

#     return train_data, train_labels


def build_model():
    print("Building model...")
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
    # train_model.summary()

    return train_model


def create_generator(train_data, train_labels):
    train_datagen = ImageDataGenerator()
    generator = train_datagen.flow(train_data, train_labels)
    while 1:
        x_batch, y_batch = generator.next()
        # print("y_batch", y_batch)
        yield ([x_batch, y_batch], [y_batch, x_batch])



def k_fold_validation(train_data, train_labels, num_folds):
    print("Running k-fold validation...")
    fold_len = train_data.shape[0] // num_folds
    # print("fold_len", fold_len)

    checkpointer = ModelCheckpoint(filepath='CapsNet.h5',
                                   monitor='val_capsnet_acc',
                                   save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_capsnet_acc', patience=4)

    results = []
    for fold in range(num_folds):
        model = build_model()
        print("++++++++++++++++++++\nProcessing fold {}...\n++++++++++++++++++++"
              .format(fold + 1))
        val_data = train_data[fold * fold_len:(fold + 1) * fold_len]
        val_data = np.expand_dims(val_data, axis=3)
        # val_data = np.reshape(val_data, (fold_len, 1, val_data.shape[1],
        #                                  val_data.shape[2]))
        val_labels = train_labels[fold * fold_len:(fold + 1) * fold_len]

        partial_train_data = np.concatenate((train_data[:fold * fold_len],
                                             train_data[(fold + 1) * fold_len:]))
        partial_train_data = np.expand_dims(partial_train_data, axis=3)
        # partial_train_data = np.reshape(partial_train_data,
        #                                 (train_data.shape[0] - fold_len, 1,
        #                                  partial_train_data.shape[1],
        #                                  partial_train_data.shape[2]))
        partial_train_labels = np.concatenate((train_labels[:fold * fold_len],
                                               train_labels[(fold + 1) * fold_len:]))

        print("Training data shape: {}".format(partial_train_data.shape))
        print("Training Labels shape: {}".format(partial_train_labels.shape))
        print("Validation data shape: {}".format(val_data.shape))
        print("Validation Labels shape: {}".format(val_labels.shape))
        train_gen = create_generator(partial_train_data, partial_train_labels)
        val_gen = create_generator(val_data, val_labels)
        hst = model.fit_generator(train_gen,
                                  # validation_data=val_gen,
                                  validation_data=([val_data, val_labels],
                                                   [val_labels, val_data]),
                                  steps_per_epoch=72,
                                  validation_steps=24,
                                  epochs=15,
                                  verbose=1,
                                  callbacks=[early_stopping,
                                             checkpointer])
        results.append(hst.history)

    return results


def plt_history(results):
    # val_acc val_loss in the same figure
    # train_acc train_loss in the same figure
    clr = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    num_epochs = np.arange(1, len(results[0]['val_capsnet_acc']) + 1)
    # Plot the val_capsnet_acc vs val_capsnet_loss
    for i, history in enumerate(results):
        plt.plot(num_epochs, history["val_capsnet_acc"], clr[i])
        plt.plot(num_epochs, history["val_capsnet_loss"], clr[i])
    plt.xlabel("Epoch")
    plt.ylabel("Validation CapsNet Acc/Loss")
    plt.savefig("val_capsnet_acc-loss.png")
    # Plot the capsnet_acc vs capsnet_loss
    plt.figure(2)
    for i, history in enumerate(results):
        plt.plot(num_epochs, history["capsnet_acc"], clr[i])
        plt.plot(num_epochs, history["capsnet_loss"], clr[i])
    plt.xlabel("Epoch")
    plt.ylabel("Training CapsNet Acc/Loss")
    plt.savefig("capsnet_acc-loss.png")
    


def main():
    train_data, train_labels = prepare_data()
    # print(train_data.shape)
    # print(train_data[:10])
    num_folds = 5
    k_fold_results = k_fold_validation(train_data, train_labels,
                                       num_folds)
    # print("Results per fold:", results_per_fold)
    # print("Mean of results:", mean)
    plt_history(k_fold_results)


if __name__ == '__main__':
    main()

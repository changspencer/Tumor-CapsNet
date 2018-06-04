import h5py
import glob
import random
import numpy as np
from keras.utils import to_categorical
from skimage import transform


def prepare_data():
    train_data, train_labels = [], []
    folders = ['RawData/brainTumorDataPublic_1766',
               'RawData/brainTumorDataPublic_7671532',
               'RawData/brainTumorDataPublic_15332298',
               'RawData/brainTumorDataPublic_22993064']
    for fold in folders:
        list_files = glob.glob(fold + "/*.mat")
        random.shuffle(list_files)
        print("Getting files from {}...".format(fold))
        for file in list_files:
            with h5py.File(file) as f:
                img = f["cjdata/image"]
                mask = f["cjdata/tumorMask"]
                train_labels.append(int(f["cjdata/label"][0]))
                img = np.array(img)
                mask = np.array(mask)
                # Normalize to 0.0 to 1.0
                img = img * (1 / np.max(np.max(img)))
                seg_img = np.multiply(img, mask)
                seg_img = transform.resize(seg_img, (64, 64))
                train_data.append(seg_img)
    train_data = np.asarray(train_data)
    print("train_data.shape = ", train_data.shape)
    train_labels = np.asarray(train_labels)
    train_labels = to_categorical(train_labels, num_classes=4)
    train_labels = train_labels[:, 1:]
    print("train_labels.shape = ", train_labels.shape)


prepare_data()

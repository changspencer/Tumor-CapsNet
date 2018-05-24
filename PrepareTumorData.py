import h5py
import sys
import os
import glob
import re
import random
import numpy as np
from matplotlib import pyplot as plt


# Constants for the tumor labels
meningioma = "1_meningioma"
glioma = "2_glioma"
pituitary = "3_pituitary"
folders = (meningioma, glioma, pituitary)


def splitMatFile(fold_name, train_fold, test_fold, test_split):
    '''
    Takes in a folder of .mat files (v7.3+) saves the image in its appropriate
    training or test folder, given its label and a provided train/test split
    '''
    list_files = glob.glob(fold_name + "/*.mat")
    random.shuffle(list_files)
    split_idx = len(list_files) * test_split
    for i, file in enumerate(list_files):
        dest_fold = train_fold if i < split_idx else test_fold
        with h5py.File(file) as f:
            match = re.search("(?:[a-zA-Z0-9_\/]*)[0-9]+(?=[.]mat)", file)
            fig_name = match.group(0) + ".png"
            img = f["cjdata/image"]
            mask = f["cjdata/tumorMask"]
            label = int(f["cjdata/label"][0])
            print("File: {} --- Label: {} --- Folder: {}"
                  .format(fig_name, folders[label - 1], dest_fold))
            img = np.array(img)
            mask = np.array(mask)
            # Normalize to 0.0 to 1.0
            img = img * (1 / np.max(np.max(img)))
            seg_img = np.multiply(img, mask)
            plt.imsave(dest_fold + '/' + folders[label - 1] + '/' + fig_name,
                       seg_img, cmap="gray")


def clearDataFolders(fold_name):
    '''
    Takes in a folder name to search for .png files.
    ***CAUTION: This targets all PNG files containing preceding digits***
    '''
    list_files = glob.glob(fold_name + "/*.png")
    print("Folder {} has {} .png files".format(fold_name, len(list_files)))
    count = 0
    for file in list_files:
        # print("File found: {}".format(file))
        match = re.search("[0-9]+.png", file)
        if match is not None:
            count += 1
            # print("Deleting {}: {}".format(count, match.group(0)))
            os.remove(fold_name + '/' + match.group(0))
    print("Removed {} .png files from folder {}".format(count, fold_name))


if __name__ == "__main__":
    print("Clearing the training folders...")
    clearDataFolders('train/' + meningioma)
    clearDataFolders('train/' + glioma)
    clearDataFolders('train/' + pituitary)
    print("Clearing the test folders...")
    clearDataFolders('test/' + meningioma)
    clearDataFolders('test/' + glioma)
    clearDataFolders('test/' + pituitary)
    print("Parsing the .mat files now....")
    splitMatFile("RawData/brainTumorDataPublic_1766", "train", "test", 0.25)

    print("Done parsing the .mat files. Enjoy the results!")

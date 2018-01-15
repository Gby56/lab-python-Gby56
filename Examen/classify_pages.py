# -*- coding: utf-8 -*-
"""
G. MARQUET - 2018
"""

import argparse
import logging
import os
import shutil
import multiprocessing
import sys
import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics, neighbors, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC



# default sub-resolution
IMG_FEATURE_SIZE = (8,8)

# Setup logging
logger = logging.getLogger('cluster_images.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)


def load_image_list(list):
    all_df = pd.read_csv(os.path.dirname(os.path.realpath(list))+"/"+list, sep=',', names=['path', 'classes','ref'])
    # Load the image list from CSV file using pd.read_csv
    # see the doc for the option since there is no header ;
    # specify the column names :  filename , class

    if args.limit_samples:
        all_df = all_df[:args.limit_samples]

    return all_df

def extract_features(img):
    """
    Compute the subresolution of an image and return it as a feature vector

    :param img: the original image (can be color or gray)
    :type img: pillow image
    :return: pixel values of the image in subresolution
    :rtype: list of int in [0,255]

    """

    # convert color images to grey level
    gray_img = img.convert('L')
    # find the min dimension to rotate the image if needed
    min_size = min(img.size)
    if img.size[1] == min_size:
        # convert landscape  to portrait
        rotated_img = gray_img.rotate(90, expand=1)
    else:
        rotated_img = gray_img

    # reduce the image to a given size
    reduced_img = rotated_img.resize(
        IMG_FEATURE_SIZE, Image.BOX).filter(ImageFilter.SHARPEN)

    # return the values of the reduced image as features
    return [255 - i for i in reduced_img.getdata()]


def copy_to_dir(images, clusters, cluster_dir):
    """
    Move images to a directory according to their cluster name

    :param images: list of image names (path)
    :type images: list of path
    :param clusters: list of cluster values (int), such as given by cluster.labels_, associated to each image
    :type clusters: list
    :param cluster_dir: prefix path where to copy the images is a drectory corresponding to each cluster
    :type images: path
    :return: None
    """

    for img_path, cluster in zip(images, clusters):
        # define the cluster path : for example "CLUSTERS/4" if the image is in cluster 4
        clst_path = os.path.join(cluster_dir, str(cluster))
        # create the directory if it does not exists
        if not os.path.exists(clst_path):
            os.mkdir(clst_path)
        # copy the image into the cluster directory
        shutil.copy(img_path, clst_path)


def image_loader(image):
    """" Extract the features from the image"""
    my_image = Image.open(os.path.realpath(image))
    return extract_features(my_image)


if __name__ == "__main__":
    #region load everything and get features
    parser = argparse.ArgumentParser(description='Extract features, cluster images and move them to a directory')
    parser.add_argument('--images-list')
    parser.add_argument('--move-images')
    parser.add_argument('--knn', type=int)
    parser.add_argument('--svm-linear', action='store_true')
    parser.add_argument('--opt', action='store_true')
    parser.add_argument('--limit-samples', type=int, help='limit the number of samples to consider for training')
    parser.add_argument('--load-features',help='read features and class from pickle file')
    parser.add_argument('--save-features',help='save features in pickle format')

    args = parser.parse_args()
    CLUSTER_DIR = ""
    if args.move_images:
        CLUSTER_DIR = args.move_images
        # Clean up
        if os.path.exists(CLUSTER_DIR):
            shutil.rmtree(CLUSTER_DIR)
            logger.info('remove cluster directory %s' % CLUSTER_DIR)
        os.mkdir(CLUSTER_DIR)

    # find all the pages in the directory
    images_path_list = []
    data = []

    if args.load_features:
        df_features = pd.read_pickle(args.load_features)
        logger.info('Loaded {} features'.format(df_features.shape))
        if args.limit_samples:
            df_features = df_features.sample(n=args.limit_samples)

        # define X (features) and y (target)
        if 'class' in df_features.columns:
            X = df_features.drop(['class'], axis=1)
            classes = df_features['class']
            data = X
        else:
            logger.error('Can not find classes in pickle')
            sys.exit(1)


    if args.images_list:
        SOURCE_IMG_DIR = args.images_list
        all_df = load_image_list(SOURCE_IMG_DIR)
        images_path_list = all_df.path
        classes = all_df.classes
        ref = all_df.ref

        if images_path_list.size == 0:
            logger.warning("Did not found any jpg image in %s" % args.images_dir)
            sys.exit(0)

        # Multi-core feature extraction.
        data = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(image_loader)(image) for image in tqdm(images_path_list))

    if not data.size>0:
        logger.error("Could not extract any feature vector")
        sys.exit(1)


    # convert to np array (default format for scikit-learn)
    X = np.array(data)
    #endregion
    logger.info("Running classifier")

    if args.save_features:
        df_features = pd.DataFrame(X)
        df_features['class'] = classes
        df_features.to_pickle(args.save_features)
        logger.info('Saved {} features and class to {}'.format(df_features.shape,args.save_features))

    # in the directory corresponding to its cluster

    if args.knn:
        logger.info("KNN SELECTED")
        X_train, X_temp, y_train, y_temp = train_test_split(X, classes, test_size=0.4)
        X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
        # create KNN classifier with args.nearest_neighbors as a parameter
        logger.info('Use kNN classifier with k = {}'.format(args.knn))
        knn = neighbors.KNeighborsClassifier(args.knn, n_jobs=-1)

        knn.fit(X_train,y_train)

        prediction_knn = knn.predict(X_test)
        print(metrics.classification_report(y_test, prediction_knn))
        print(str(metrics.accuracy_score(y_test, prediction_knn, normalize=True)) + " for k = " +str(args.knn))

        if args.opt:
            accuracies = []
            for kk in range(1,10):
                # create KNN classifier with args.nearest_neighbors as a parameter
                knn = neighbors.KNeighborsClassifier(kk, n_jobs=-1)

                knn.fit(X_train, y_train)

                prediction_knn = knn.predict(X_test)
                print(str(metrics.accuracy_score(y_test, prediction_knn, normalize=True)) + ' for k = ' + str(kk))
                accuracies.append(metrics.accuracy_score(y_test, prediction_knn, normalize=True))
            print("Best was " + str(max(accuracies)) +" for k = " + str(np.argmax(accuracies) + 1))

    elif args.svm_linear:
        # Question 10
        clf = svm.LinearSVC()
        clf_name = "LinearSVM"

        print("Linear Kernel \n")

        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size=0.2, shuffle=False)

        clf = SVC(kernel='linear')
        print("Training Linear SVC...")
        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)
        print("TEST set : ")
        print(metrics.classification_report(y_test, pred))
        print("Accuracy : " + str(metrics.accuracy_score(y_test,pred,normalize=True)))

        pred = clf.predict(X_train)
        print("TRAIN set : ")
        print(metrics.classification_report(y_train, pred))
        print("Accuracy : " + str(metrics.accuracy_score(y_train,pred,normalize=True)))


    if not args.images_list:
        logger.info(msg="Cluster image directory was not specified, exiting")
        sys.exit(1)

    # copy_to_dir(images_path_list, kmeans_model.labels_, CLUSTER_DIR)

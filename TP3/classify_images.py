# -*- coding: utf-8 -*-
"""
Classify digit images

C. Kermorvant - 2017
"""

import pickle
import argparse
import logging
import time
import sys
import multiprocessing
import os
import numpy as np
import pandas as pd


from tqdm import tqdm
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from sklearn import svm, metrics, neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed


# Setup logging
logger = logging.getLogger('classify_images.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)


def image_loader(entry):
    path = entry[0]
    clas = entry[1]
    """" Extract the features from the image"""
    my_image = Image.open(os.path.dirname(os.path.realpath(__file__)) + "/" + path)

    """
    code to show features as image
    onche  = extract_features_subresolution(my_image)
    B = np.asarray(onche).reshape((8,8))
    img2 = Image.fromarray(np.uint8(B),'L')
    img2.save("onche3.png")
    """

    return extract_features_subresolution(my_image,clas)

def getOptimalK(X, classes):
    print("-------------------------")
    print("COMPUTING OPTIMAL K VALUE")
    # creating odd list of K for KNN
    ks = list(range(1, 10))

    # empty list that will hold cv scores
    cv_scores = []

    # perform 10-fold cross validation
    for k in ks:
        knn2 = neighbors.KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn2, X, classes, cv=5, scoring='accuracy',n_jobs=-1, verbose=2)
        print("Cross validation scores for k = " + str(k) + " out of 9")
        print(scores)
        print("Average of the scores : " + str(scores.mean()))
        cv_scores.append(scores.mean())

    print(cv_scores)

    # changing to misclassification error
    MSE = [1 - x for x in cv_scores]

    # determining best k
    optimal_k = ks[MSE.index(min(MSE))]
    print("The optimal number of neighbors is %d" % optimal_k)
    #optimal is 6
    return optimal_k

def getTrainingError(kvalue, X_test, y_test, X_train, y_train):
    print("-----------------" + " K = " +str(kvalue)+ " ---------------------")

    if __name__ == '__main__':

        knn = neighbors.KNeighborsClassifier(n_neighbors=kvalue, p=2, metric='euclidean',  n_jobs=-1)

        # Do Training
        t0 = time.time()
        knn.fit(X_train, y_train)
        logger.info("Training done in %0.3fs" % (time.time() - t0))

        # Do testing
        logger.info("Testing Classifier on test & training sets")
        t0 = time.time()
        predicted = knn.predict(X_test)

        # Print score produced by metrics.classification_report and metrics.accuracy_score
        logger.info("Testing  done on test set in %0.3fs" % (time.time() - t0))

        print("Report for prediction on test set")
        print(metrics.classification_report(y_test, predicted))
        print("Accuracy score : " + str(metrics.accuracy_score(y_test, predicted, normalize=True)) + " with k = " + str(kvalue))

        t0 = time.time()
        predicted2 = knn.predict(X_train)
        logger.info("Testing  done on training set in %0.3fs" % (time.time() - t0))

        print("Report for prediction on training set")
        print(metrics.classification_report(y_train, predicted2))
        print("Accuracy score : " + str(metrics.accuracy_score(y_train, predicted2, normalize=True)) + " with k = " + str(kvalue))

    return

class FeatureClass:
    def __init__(self,features,clas):
        self.features = features
        self.clas = clas
    def __str__(self):
        return(str(self.clas) + " | " + str(self.features))

def extract_features_subresolution(img, clas, img_feature_size=(8, 8)):
    """
    Compute the subresolution of an image and return it as a feature vector

    :param img: the original image (can be color or gray)
    :type img: pillow image
    :return: pixel values of the image in subresolution
    :rtype: list of int in [0,255]

    """

    # convert color images to grey level
    gray_img = img.convert('L')

    # reduce the image to a given size
    reduced_img = gray_img.resize(
        img_feature_size, Image.BOX).filter(ImageFilter.SHARPEN)


    # return the values of the reduced image as features
    return FeatureClass([i for i in reduced_img.getdata()],clas)


if __name__ == "__main__":
    #region define parser
    parser = argparse.ArgumentParser(
        description='Extract features, train a classifier on images and test the classifier')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--images-list',
                             help='file containing the image path and image class, one per line, comma separated')
    input_group.add_argument('--load-features', help='read features and class from pickle file')
    parser.add_argument('--save-features', help='save features in pickle format')
    parser.add_argument('--limit-samples', type=int, help='limit the number of samples to consider for training')
    classifier_group = parser.add_mutually_exclusive_group(required=True)
    classifier_group.add_argument('--nearest-neighbors', type=int)
    classifier_group.add_argument('--features-only', action='store_true',
                                  help='only extract features, do not train classifiers')
    args = parser.parse_args()
    #endregion


    if args.load_features:
        all = []
        # read features from to_pickle
        with (open(args.load_features, "rb")) as openfile:
                try:
                    all.append(pickle.load(openfile))
                except EOFError:print("error")

        loaded_features = []
        all_df_temp = []

        for entry in range(0,len(all[0])):
            loaded_features.append(all[0][entry].features)
            all_df_temp.append(all[0][entry].clas)

        featclass = np.array(all[0])
        all_df_temp = np.array(all_df_temp)
        all_df = pd.DataFrame([])
        all_df['Clas'] = pd.Series(all_df_temp)

        logger.info("Loaded {} file".format(args.load_features))

    else:
        all_df = pd.read_csv(args.images_list, names=['Filename', 'Clas'])
        # Load the image list from CSV file using pd.read_csv
        # see the doc for the option since there is no header ;
        # specify the column names :  filename , class
        file_list = all_df.Filename

        logger.info('Loaded {} images in {}'.format(all_df.shape, args.images_list))

        # Extract the feature vector on all the pages found
        # Modify the extract_features from TP_Clustering to extract 8x8 subresolution values
        # white must be 0 and black 255
        data = []


        # Multi-core feature extraction.
        data = Parallel(n_jobs=multiprocessing.cpu_count())(
           delayed(image_loader)(entry) for entry in tqdm(all_df.values))


        if not data:
            logger.error("Could not extract any feature vector or class")
            sys.exit(1)

        # convert to np.array
        featclass = np.array(data)

    # save features
    if args.save_features:
        # convert X to dataframe with pd.DataFrame and save to pickle with to_pickle

        with open(args.save_features, 'wb') as f:
            pickle.dump(featclass, f)

        #to_save.to_pickle(args.save_features)
        logger.info('Saved {} features and class to {}'.format(featclass.shape, args.save_features))


    if args.features_only:
        logger.info('No classifier to train, exit')
        sys.exit()

    X = []
    Y = []
    for ft in featclass:
        X.append(ft.features)
        Y.append(ft.clas)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    '''
    B = np.asarray(X_train[32006]).reshape((8, 8))
    img2 = Image.fromarray(np.uint8(B), 'L')
    lel = y_train[32006]
    lol = file_list[32006]
    img2.save("onchenew.png")
    '''

    #Use train_test_split to create train/test split
    logger.info("Train set size is {}".format(len(X_train)))
    logger.info("Test set size is {}".format(len(X_test)))

    if args.nearest_neighbors:
        # create KNN classifier with args.nearest_neighbors as a parameter
        logger.info('Use kNN classifier with k = {}'.format(args.nearest_neighbors))
        knn = neighbors.KNeighborsClassifier(args.nearest_neighbors, n_jobs=-1)

        # TRY WITH K=1
        getTrainingError(args.nearest_neighbors, X_test, y_test, X_train, y_train)

        # TRY ON RANGE OF K VALUES
        '''ks = list(range(2, 10))
        for k in ks:
            getTrainingError(k, X_test, y_test, X_train, y_train)'''

        # getOptimalK(X_train, y_train)
    else:
        logger.error('No classifier specified')
        sys.exit()
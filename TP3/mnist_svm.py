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
import matplotlib.pyplot as plt
import math



from tqdm import tqdm
from PIL import Image, ImageFilter
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Setup logging
logger = logging.getLogger('mnist_svm.py')
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

class FeatureClass:
    def __init__(self,features,clas):
        self.features = features
        self.clas = clas
    def __str__(self):
        return(str(self.clas) + " | " + str(self.features))

if __name__ == "__main__":
    #region define parser
    parser = argparse.ArgumentParser(
        description='Extract features, train a classifier on images and test the classifier')
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--images-list', help='file containing the image path and image class, one per line, comma separated')
    input_group.add_argument('--load-features', help='read features and class from pickle file')

    parser.add_argument('--save-features', help='save features in pickle format')
    parser.add_argument('--limit-samples', type=int, help='limit the number of samples to consider for training')
    parser.add_argument('--features-only', action='store_true', help='only extract features, do not train classifiers')

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--kernel', type=str)
    mode.add_argument('--usps', action='store_true')
    args = parser.parse_args()
    #endregion

    if args.usps:
        all = []
        # read features from to_pickle
        with (open('feat.pkl', "rb")) as openfile:
            try:
                all.append(pickle.load(openfile))
            except EOFError:
                print("error")

        # loaded_features = []
        # all_df_temp = []

        featclass = np.array(all[0])

        '''
        for entry in range(0,len(all[0])):
            #loaded_features.append(all[0][entry].features)
            all_df_temp.append(all[0][entry].clas)


        all_df_temp = np.array(all_df_temp)
        all_df = pd.DataFrame([])
        all_df['Clas'] = pd.Series(all_df_temp)
        '''

        logger.info("Loaded {} file".format('feat.pkl'))

        allusps = []
        # read features from to_pickle
        with (open('usps.pkl', "rb")) as openfile:
            try:
                allusps.append(pickle.load(openfile))
            except EOFError:
                print("error")

        # loaded_features = []
        # all_df_temp = []

        featclassusps = np.array(allusps[0])

        '''
        for entry in range(0,len(all[0])):
            #loaded_features.append(all[0][entry].features)
            all_df_temp.append(all[0][entry].clas)


        all_df_temp = np.array(all_df_temp)
        all_df = pd.DataFrame([])
        all_df['Clas'] = pd.Series(all_df_temp)
        '''

        logger.info("Loaded {} file".format('usps.pkl'))

        # region load data in variables
        X = []
        Y = []
        np.random.shuffle(featclass)
        for ft in featclass:
            X.append(ft.features)
            Y.append(ft.clas)

        # endregion

        # region load data in variables
        Xusps = []
        Yusps = []
        np.random.shuffle(featclassusps)
        for ft in featclassusps:
            Xusps.append(ft.features)
            Yusps.append(ft.clas)

        # endregion

        print("RBF Kernel training on MNIST and testing on USPS\n")

        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
        Xusps = StandardScaler(with_mean=True, with_std=True).fit_transform(Xusps)

        svc = SVC(kernel='rbf', gamma=0.05, C=10)
        #parameters = {'gamma':[0.05,0.1,0.5],'C': [1, 10]}
        #clf = GridSearchCV(svc, parameters,n_jobs=-1,verbose=1)

        svc.fit(X, Y)

        print("Predicting USPS features to classes...")
        pred = svc.predict(Xusps)

        for p in range(0,len(pred)):
            print(str(pred[p]) + "  |  " + str(Yusps[p]))
            if(p%50==0):
                print("pred|real")

        print("USPS set : ")
        print(print(metrics.classification_report(Yusps, pred)))
        print("Accuracy : " + str(accuracy_score(Yusps,pred,normalize=True)))

    if args.load_features:
        all = []
        # read features from to_pickle
        with (open(args.load_features, "rb")) as openfile:
            try:
                all.append(pickle.load(openfile))
            except EOFError:
                print("error")

        # loaded_features = []
        # all_df_temp = []

        featclass = np.array(all[0])

        '''
        for entry in range(0,len(all[0])):
            #loaded_features.append(all[0][entry].features)
            all_df_temp.append(all[0][entry].clas)


        all_df_temp = np.array(all_df_temp)
        all_df = pd.DataFrame([])
        all_df['Clas'] = pd.Series(all_df_temp)
        '''

        logger.info("Loaded {} file".format(args.load_features))

    elif args.images_list:
        all_df = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + args.images_list, names=['Filename', 'Clas'], sep=' ')
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

    if args.save_features:
        # convert X to dataframe with pd.DataFrame and save to pickle with to_pickle

        with open(args.save_features, 'wb') as f:
            pickle.dump(featclass, f)

        #to_save.to_pickle(args.save_features)
        logger.info('Saved {} features and class to {}'.format(featclass.shape, args.save_features))

    if args.features_only:
        logger.info('No classifier to train, exit')
        sys.exit()

    #region load data in variables
    X = []
    Y = []
    np.random.shuffle(featclass)
    for ft in featclass:
        X.append(ft.features)
        Y.append(ft.clas)

    #endregion

    if args.limit_samples:
        X = X[:args.limit_samples]
        Y = Y[:args.limit_samples]

    #region create image from feature array to check
    '''
    B = np.asarray(X[1206]).reshape((8, 8))
    img2 = Image.fromarray(np.uint8(B), 'L')
    lel = Y[1206]
    lol = file_list[1206]
    img2.save("onchenew.png")
    '''
    #endregion



    if args.kernel == 'linear':
        print("Linear Kernel \n")

        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)
        print("TEST set : ")
        print(print(metrics.classification_report(y_test, pred)))
        print("Accuracy : " + str(accuracy_score(y_test,pred,normalize=True)))

        pred = clf.predict(X_train)
        print("TRAIN set : ")
        print(print(metrics.classification_report(y_train, pred)))
        print("Accuracy : " + str(accuracy_score(y_train,pred,normalize=True)))

    elif args.kernel == 'RBF':
        print("RBF Kernel \n")

        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

        svc = SVC(kernel='rbf')
        parameters = {'gamma':[0.05,0.1,0.5],'C': [1, 10]}
        clf = GridSearchCV(svc, parameters,n_jobs=-1,verbose=1)
        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)
        print("TEST set : ")
        print(print(metrics.classification_report(y_test, pred)))
        print("Accuracy : " + str(accuracy_score(y_test,pred,normalize=True)))
        print("Best params : " + str(clf.best_params_) + " using " + str(clf.n_splits_) + " folds with a best score of " + str(clf.best_score_))

    else:
        logger.error('No classifier specified')
        sys.exit()
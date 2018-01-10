# -*- coding: utf-8 -*-
"""
Classify text

G.Marquet 2017
"""

import pickle
import logging
import time
import sys
import multiprocessing
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns


from tqdm import tqdm
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from sklearn import svm, metrics, neighbors, linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from joblib import Parallel, delayed

# region Setup logging
logger = logging.getLogger('classify_images.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)


# endregion


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
        scores = cross_val_score(knn2, X, classes, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
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
    # optimal is 6
    return optimal_k


def getOptimalK2(X, Y):
    print("-------------------------")
    print("COMPUTING OPTIMAL K VALUE, FROM 1 TO 10")

    # Create three sets : train set (60%), validation set (20%) and test set (20%), using twice train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    # empty list that will hold cv scores
    cv_scores = []
    # cv_scores = [[0.91800000000000004, 1.0, 0.92174999999999996], [0.91308333333333336, 0.96272222222222226, 0.91333333333333333], [0.92500000000000004, 0.96136111111111111, 0.92691666666666672], [0.92549999999999999, 0.95372222222222225, 0.92608333333333337], [0.9291666666666667, 0.95138888888888884, 0.92966666666666664], [0.92666666666666664, 0.94808333333333328, 0.92700000000000005], [0.92608333333333337, 0.94611111111111112, 0.92941666666666667], [0.92474999999999996, 0.94405555555555554, 0.92800000000000005], [0.92483333333333329, 0.94216666666666671, 0.92633333333333334]]

    # TRY ON RANGE OF K VALUES
    ks = list(range(1, 10, 2))
    for k in ks:
        cv_scores.append(getTrainingValidationError(k, X_test, y_test, X_train, y_train, X_val, y_val))

    print(cv_scores)
    ks = range(1, 10, 2)

    plt.plot(ks, cv_scores)
    plt.ylabel('Accuracy')
    plt.xlabel('K value')
    plt.title('Accuracy on testing & training set for different k values')
    plt.show()

    print(
        "K = 5 seems to be the best value because of the high accuracy score & minimum error between the predictions on the test and validation set")
    return


def getTrainingError(kvalue, X_test, y_test, X_train, y_train, logging):
    print("-------------------- " + "| K = " + str(kvalue) + "| ---------------------")

    # sorry for the logging variable but idc and it's easier
    knn = neighbors.KNeighborsClassifier(n_neighbors=kvalue, n_jobs=-1)

    # Do Training
    t0 = time.time()
    knn.fit(X_train, y_train)
    if logging:
        logger.info("Training done in %0.3fs" % (time.time() - t0))

        # Do testing
        logger.info("Testing Classifier on test & training sets")
    t0 = time.time()
    predicted = knn.predict(X_test)

    if logging:
        # Print score produced by metrics.classification_report and metrics.accuracy_score
        logger.info("Testing  done on test set in %0.3fs" % (time.time() - t0))

        print("Report for prediction on test set")
        print(metrics.classification_report(y_test, predicted))
    acc_test = metrics.accuracy_score(y_test, predicted, normalize=True)

    if logging:
        print("Accuracy score : " + str(metrics.accuracy_score(y_test, predicted, normalize=True)) + "\n")

    t0 = time.time()
    predicted2 = knn.predict(X_train)

    if logging:
        logger.info("Testing  done on training set in %0.3fs" % (time.time() - t0))

        print("Report for prediction on training set")
        print(metrics.classification_report(y_train, predicted2))
    acc_train = knn.score(X_train, y_train)

    if logging:
        print("Accuracy score : " + str(metrics.accuracy_score(y_train, predicted2, normalize=True)) + "\n")

    return [acc_test, acc_train]


def getTrainingValidationError(kvalue, X_test, y_test, X_train, y_train, X_val, y_val):
    print("-----------------" + " K = " + str(kvalue) + " ---------------------")

    knn = neighbors.KNeighborsClassifier(n_neighbors=kvalue, p=2, metric='euclidean', n_jobs=-1)

    # Do Training
    t0 = time.time()
    knn.fit(X_train, y_train)
    print("Training done in %0.3fs" % (time.time() - t0) + "\n")

    # ////////////////////ACCURACY ON TEST SET
    print("////Test set")
    t0 = time.time()
    predicted = knn.predict(X_test)
    print("Test set done in %0.3fs" % (time.time() - t0))
    acc_test = metrics.accuracy_score(y_test, predicted, normalize=True)
    print("Accuracy score : " + str(metrics.accuracy_score(y_test, predicted, normalize=True)) + "\n")

    # ////////////////////ACCURACY ON TRAINING SET
    print("////Training set")
    t0 = time.time()
    predicted2 = knn.predict(X_train)
    print("Training set done in %0.3fs" % (time.time() - t0))
    acc_train = metrics.accuracy_score(y_train, predicted2, normalize=True)
    print("Accuracy score : " + str(metrics.accuracy_score(y_train, predicted2, normalize=True)) + "\n")

    # ////////////////////ACCURACY ON VALIDATION SET
    print("////Validation set")
    t0 = time.time()
    predicted3 = knn.predict(X_val)
    print("Validation set done in %0.3fs" % (time.time() - t0))
    acc_val = metrics.accuracy_score(y_val, predicted3, normalize=True)
    print("Accuracy score : " + str(metrics.accuracy_score(y_val, predicted3, normalize=True)) + "\n")

    # [[0.91800000000000004, 1.0, 0.92174999999999996], [0.91308333333333336, 0.96272222222222226, 0.91333333333333333], [0.92500000000000004, 0.96136111111111111, 0.92691666666666672], [0.92549999999999999, 0.95372222222222225, 0.92608333333333337], [0.9291666666666667, 0.95138888888888884, 0.92966666666666664], [0.92666666666666664, 0.94808333333333328, 0.92700000000000005], [0.92608333333333337, 0.94611111111111112, 0.92941666666666667], [0.92474999999999996, 0.94405555555555554, 0.92800000000000005], [0.92483333333333329, 0.94216666666666671, 0.92633333333333334]]
    return [acc_test, acc_train, acc_val]


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
    return FeatureClass([i for i in reduced_img.getdata()], clas)

def simpleMnb(alpha, max_features):

    #region Transforming text to features with Countvectorizer
    vectorizer = CountVectorizer(max_features=max_features)
    vectorizer.fit(X_train)
    X_train_counts = vectorizer.transform(X_train)
    X_test_counts = vectorizer.transform(X_test)
    X_dev_counts = vectorizer.transform(X_dev)
    #endregion

    #region Countvectorizer on MnB
    mnb = MultinomialNB(alpha=alpha)
    mnb.fit(X_train_counts,y_train)
    y_pred_train = mnb.predict(X_train_counts)
    y_pred_test = mnb.predict(X_test_counts)
    y_pred_dev = mnb.predict(X_dev_counts)



    #print("Train set Accuracy score : " + str(metrics.accuracy_score(y_train, y_pred_train)))
    #print("Test set Accuracy score : " + str(metrics.accuracy_score(y_test, y_pred_test)))
    #print("Dev set Accuracy score : " + str(metrics.accuracy_score(y_dev, y_pred_dev)))
    #endregion
    return metrics.accuracy_score(y_test, y_pred_test)

def MnBTuning(values):
    #print("TESTING FOR alpha = " + str(values[0]) + " & max_features = " + str(2 ** values[1]))
    return {'alpha':values[0],'max_features':2**values[1],'accuracy':simpleMnb(values[0],2**values[1])}

if __name__ == "__main__":

    #region SETUP
    #region reading data
    all_df = pd.read_csv("LeMonde2003.csv",sep='\t', header=0)

    all_df.dropna(axis=0,how='any',inplace=True)

    logger.info("Loaded {} file".format("LeMonde2003.csv"))


    all_df = all_df[all_df['category'].isin(['ENT','INT','ART','SOC','FRA','SPO','LIV','TEL','UNE'])]


    logger.info("Kept values for categories : ['ENT','INT','ART','SOC','FRA','SPO','LIV','TEL','UNE']")

    sns.countplot(x="category", data=all_df)
    plt.show()

    #endregion

    #region splitting data
    X_train, X_temp, y_train, y_temp = train_test_split(all_df.text, all_df.category, test_size=0.4)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
    #endregion


    #endregion

    # errorsSimpleMnb=[{'max_features': 4096, 'alpha': 0.0, 'accuracy': 0.7768937510359688}, {'max_features': 8192, 'alpha': 0.0, 'accuracy': 0.79595557765622416}, {'max_features': 16384, 'alpha': 0.0, 'accuracy': 0.80706116360019886}, {'max_features': 32768, 'alpha': 0.0, 'accuracy': 0.80275153323388038}, {'max_features': 65536, 'alpha': 0.0, 'accuracy': 0.80043096303663186}, {'max_features': 4096, 'alpha': 0.10000000000000001, 'accuracy': 0.7767279960218797}, {'max_features': 8192, 'alpha': 0.10000000000000001, 'accuracy': 0.79297198740261898}, {'max_features': 16384, 'alpha': 0.10000000000000001, 'accuracy': 0.80573512348748555}, {'max_features': 32768, 'alpha': 0.10000000000000001, 'accuracy': 0.80987899883971493}, {'max_features': 65536, 'alpha': 0.10000000000000001, 'accuracy': 0.81667495441737115}, {'max_features': 4096, 'alpha': 0.20000000000000001, 'accuracy': 0.7768937510359688}, {'max_features': 8192, 'alpha': 0.20000000000000001, 'accuracy': 0.79230896734626222}, {'max_features': 16384, 'alpha': 0.20000000000000001, 'accuracy': 0.80225426819161283}, {'max_features': 32768, 'alpha': 0.20000000000000001, 'accuracy': 0.80706116360019886}, {'max_features': 65536, 'alpha': 0.20000000000000001, 'accuracy': 0.81418862920603352}, {'max_features': 4096, 'alpha': 0.30000000000000004, 'accuracy': 0.77656224100779048}, {'max_features': 8192, 'alpha': 0.30000000000000004, 'accuracy': 0.79164594728990556}, {'max_features': 16384, 'alpha': 0.30000000000000004, 'accuracy': 0.80109398309298852}, {'max_features': 32768, 'alpha': 0.30000000000000004, 'accuracy': 0.80639814354384221}, {'max_features': 65536, 'alpha': 0.30000000000000004, 'accuracy': 0.81335985413558765}, {'max_features': 4096, 'alpha': 0.40000000000000002, 'accuracy': 0.7767279960218797}, {'max_features': 8192, 'alpha': 0.40000000000000002, 'accuracy': 0.79098292723354879}, {'max_features': 16384, 'alpha': 0.40000000000000002, 'accuracy': 0.80043096303663186}, {'max_features': 32768, 'alpha': 0.40000000000000002, 'accuracy': 0.80590087850157466}, {'max_features': 65536, 'alpha': 0.40000000000000002, 'accuracy': 0.8140228741919443}, {'max_features': 4096, 'alpha': 0.5, 'accuracy': 0.7767279960218797}, {'max_features': 8192, 'alpha': 0.5, 'accuracy': 0.79065141720537047}, {'max_features': 16384, 'alpha': 0.5, 'accuracy': 0.7997679429802752}, {'max_features': 32768, 'alpha': 0.5, 'accuracy': 0.80507210343112878}, {'max_features': 65536, 'alpha': 0.5, 'accuracy': 0.81468589424830096}, {'max_features': 4096, 'alpha': 0.60000000000000009, 'accuracy': 0.7767279960218797}, {'max_features': 8192, 'alpha': 0.60000000000000009, 'accuracy': 0.78982264213492459}, {'max_features': 16384, 'alpha': 0.60000000000000009, 'accuracy': 0.79910492292391844}, {'max_features': 32768, 'alpha': 0.60000000000000009, 'accuracy': 0.80556936847339633}, {'max_features': 65536, 'alpha': 0.60000000000000009, 'accuracy': 0.81452013923421185}, {'max_features': 4096, 'alpha': 0.70000000000000007, 'accuracy': 0.77656224100779048}, {'max_features': 8192, 'alpha': 0.70000000000000007, 'accuracy': 0.78915962207856782}, {'max_features': 16384, 'alpha': 0.70000000000000007, 'accuracy': 0.79927067793800766}, {'max_features': 32768, 'alpha': 0.70000000000000007, 'accuracy': 0.80639814354384221}, {'max_features': 65536, 'alpha': 0.70000000000000007, 'accuracy': 0.81452013923421185}, {'max_features': 4096, 'alpha': 0.80000000000000004, 'accuracy': 0.77656224100779048}, {'max_features': 8192, 'alpha': 0.80000000000000004, 'accuracy': 0.78866235703630039}, {'max_features': 16384, 'alpha': 0.80000000000000004, 'accuracy': 0.79943643295209676}, {'max_features': 32768, 'alpha': 0.80000000000000004, 'accuracy': 0.80722691861428808}, {'max_features': 65536, 'alpha': 0.80000000000000004, 'accuracy': 0.81435438422012263}, {'max_features': 4096, 'alpha': 0.90000000000000002, 'accuracy': 0.77656224100779048}, {'max_features': 8192, 'alpha': 0.90000000000000002, 'accuracy': 0.78866235703630039}, {'max_features': 16384, 'alpha': 0.90000000000000002, 'accuracy': 0.79893916790982922}, {'max_features': 32768, 'alpha': 0.90000000000000002, 'accuracy': 0.80689540858610975}, {'max_features': 65536, 'alpha': 0.90000000000000002, 'accuracy': 0.81435438422012263}]
    # [{'max_features': 32768, 'alpha': 0.0001, 'accuracy': 0.81037626388198247},
    #  {'max_features': 65536, 'alpha': 0.0001, 'accuracy': 0.81269683407923088},
    #  {'max_features': 131072, 'alpha': 0.0001, 'accuracy': 0.81468589424830096},
    #  {'max_features': 262144, 'alpha': 0.0001, 'accuracy': 0.81468589424830096},
    #  {'max_features': 524288, 'alpha': 0.0001, 'accuracy': 0.81468589424830096},
    #  {'max_features': 32768, 'alpha': 0.10010000000000001, 'accuracy': 0.80971324382562571},
    #  {'max_features': 65536, 'alpha': 0.10010000000000001, 'accuracy': 0.81203381402287422},
    #  {'max_features': 131072, 'alpha': 0.10010000000000001, 'accuracy': 0.8140228741919443},
    #  {'max_features': 262144, 'alpha': 0.10010000000000001, 'accuracy': 0.8140228741919443},
    #  {'max_features': 524288, 'alpha': 0.10010000000000001, 'accuracy': 0.8140228741919443},
    #  {'max_features': 32768, 'alpha': 0.2001, 'accuracy': 0.80606663351566388},
    #  {'max_features': 65536, 'alpha': 0.2001, 'accuracy': 0.81021050886789325},
    #  {'max_features': 131072, 'alpha': 0.2001, 'accuracy': 0.81319409912149843},
    #  {'max_features': 262144, 'alpha': 0.2001, 'accuracy': 0.81319409912149843},
    #  {'max_features': 524288, 'alpha': 0.2001, 'accuracy': 0.81319409912149843},
    #  {'max_features': 32768, 'alpha': 0.30010000000000003, 'accuracy': 0.80623238852975299},
    #  {'max_features': 65536, 'alpha': 0.30010000000000003, 'accuracy': 0.80971324382562571},
    #  {'max_features': 131072, 'alpha': 0.30010000000000003, 'accuracy': 0.81369136416376597},
    #  {'max_features': 262144, 'alpha': 0.30010000000000003, 'accuracy': 0.81369136416376597},
    #  {'max_features': 524288, 'alpha': 0.30010000000000003, 'accuracy': 0.81369136416376597},
    #  {'max_features': 32768, 'alpha': 0.40010000000000001, 'accuracy': 0.80590087850157466},
    #  {'max_features': 65536, 'alpha': 0.40010000000000001, 'accuracy': 0.81004475385380403},
    #  {'max_features': 131072, 'alpha': 0.40010000000000001, 'accuracy': 0.8140228741919443},
    #  {'max_features': 262144, 'alpha': 0.40010000000000001, 'accuracy': 0.8140228741919443},
    #  {'max_features': 524288, 'alpha': 0.40010000000000001, 'accuracy': 0.8140228741919443},
    #  {'max_features': 32768, 'alpha': 0.50009999999999999, 'accuracy': 0.80556936847339633},
    #  {'max_features': 65536, 'alpha': 0.50009999999999999, 'accuracy': 0.80987899883971493},
    #  {'max_features': 131072, 'alpha': 0.50009999999999999, 'accuracy': 0.81468589424830096},
    #  {'max_features': 262144, 'alpha': 0.50009999999999999, 'accuracy': 0.81468589424830096},
    #  {'max_features': 524288, 'alpha': 0.50009999999999999, 'accuracy': 0.81468589424830096},
    #  {'max_features': 32768, 'alpha': 0.60010000000000008, 'accuracy': 0.80623238852975299},
    #  {'max_features': 65536, 'alpha': 0.60010000000000008, 'accuracy': 0.81004475385380403},
    #  {'max_features': 131072, 'alpha': 0.60010000000000008, 'accuracy': 0.8140228741919443},
    #  {'max_features': 262144, 'alpha': 0.60010000000000008, 'accuracy': 0.8140228741919443},
    #  {'max_features': 524288, 'alpha': 0.60010000000000008, 'accuracy': 0.8140228741919443},
    #  {'max_features': 32768, 'alpha': 0.70010000000000006, 'accuracy': 0.80639814354384221},
    #  {'max_features': 65536, 'alpha': 0.70010000000000006, 'accuracy': 0.81037626388198247},
    #  {'max_features': 131072, 'alpha': 0.70010000000000006, 'accuracy': 0.81369136416376597},
    #  {'max_features': 262144, 'alpha': 0.70010000000000006, 'accuracy': 0.81369136416376597},
    #  {'max_features': 524288, 'alpha': 0.70010000000000006, 'accuracy': 0.81369136416376597},
    #  {'max_features': 32768, 'alpha': 0.80010000000000003, 'accuracy': 0.80656389855793142},
    #  {'max_features': 65536, 'alpha': 0.80010000000000003, 'accuracy': 0.81037626388198247},
    #  {'max_features': 131072, 'alpha': 0.80010000000000003, 'accuracy': 0.81269683407923088},
    #  {'max_features': 262144, 'alpha': 0.80010000000000003, 'accuracy': 0.81269683407923088},
    #  {'max_features': 524288, 'alpha': 0.80010000000000003, 'accuracy': 0.81269683407923088},
    #  {'max_features': 32768, 'alpha': 0.90010000000000001, 'accuracy': 0.80689540858610975},
    #  {'max_features': 65536, 'alpha': 0.90010000000000001, 'accuracy': 0.81054201889607158},
    #  {'max_features': 131072, 'alpha': 0.90010000000000001, 'accuracy': 0.81269683407923088},
    #  {'max_features': 262144, 'alpha': 0.90010000000000001, 'accuracy': 0.81269683407923088},
    #  {'max_features': 524288, 'alpha': 0.90010000000000001, 'accuracy': 0.81269683407923088}]

    combinations = []
    for a in np.arange(0.0001, 1.0, 0.1):
        for max in range(10,21):
            combinations.append([a,max])

    # Multi-core feature extraction.
    errorsSimpleMnb = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(MnBTuning)(entry) for entry in tqdm(combinations))



    maxacc = errorsSimpleMnb[0]['accuracy']
    bestalpha = 0
    numfeat = 0

    for x in errorsSimpleMnb:
        if x['accuracy'] > maxacc:
            maxacc=x['accuracy']
            bestalpha = x['alpha']
            numfeat = x['max_features']

    print(str(errorsSimpleMnb))
    print("Maximum accuracy is : " + str(maxacc) + " with alpha = " + str(bestalpha) + " and max_features = " + str(numfeat))



    #region TFID on CountVectorizer on MnB

    logger.info("TESTING TFID REPRESENTATION")
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    X_test_tf = tf_transformer.transform(X_test_counts)
    X_dev_tf = tf_transformer.transform(X_dev_counts)

    mnb = MultinomialNB()
    mnb.fit(X_train_tf,y_train)
    y_pred_train = mnb.predict(X_train_tf)
    y_pred_test = mnb.predict(X_test_tf)
    y_pred_dev = mnb.predict(X_dev_tf)


    # print(metrics.classification_report(y_train, y_pred_train))
    # print(metrics.classification_report(y_test, y_pred_test))
    # print(metrics.classification_report(y_dev, y_pred_dev))

    print("Train set Accuracy score : " + str(metrics.accuracy_score(y_train, y_pred_train)))
    print("Test set Accuracy score : " + str(metrics.accuracy_score(y_test, y_pred_test)))
    print("Dev set Accuracy score : " + str(metrics.accuracy_score(y_dev, y_pred_dev)))

    #endregion


    print("onche")

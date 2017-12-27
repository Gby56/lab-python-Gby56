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
from sklearn.cluster import KMeans
from sklearn import svm, metrics, neighbors, linear_model
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

def getOptimalK2(X, Y):
    print("-------------------------")
    print("COMPUTING OPTIMAL K VALUE, FROM 1 TO 10")

    #Create three sets : train set (60%), validation set (20%) and test set (20%), using twice train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    # empty list that will hold cv scores
    cv_scores = []
    #cv_scores = [[0.91800000000000004, 1.0, 0.92174999999999996], [0.91308333333333336, 0.96272222222222226, 0.91333333333333333], [0.92500000000000004, 0.96136111111111111, 0.92691666666666672], [0.92549999999999999, 0.95372222222222225, 0.92608333333333337], [0.9291666666666667, 0.95138888888888884, 0.92966666666666664], [0.92666666666666664, 0.94808333333333328, 0.92700000000000005], [0.92608333333333337, 0.94611111111111112, 0.92941666666666667], [0.92474999999999996, 0.94405555555555554, 0.92800000000000005], [0.92483333333333329, 0.94216666666666671, 0.92633333333333334]]


    # TRY ON RANGE OF K VALUES
    ks = list(range(1, 10, 2))
    for k in ks:
        cv_scores.append(getTrainingValidationError(k, X_test, y_test, X_train, y_train, X_val, y_val))

    print(cv_scores)
    ks = range(1, 10, 2)

    plt.plot(ks,cv_scores)
    plt.ylabel('Accuracy')
    plt.xlabel('K value')
    plt.show()

    print("K = 5 seems to be the best value because of the high accuracy score & minimum error between the predictions on the test and validation set")
    return

def getTrainingError(kvalue, X_test, y_test, X_train, y_train, logging):
    print("-------------------- " + "| K = " +str(kvalue)+ "| ---------------------")

    #sorry for the logging variable but idc and it's easier
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

    return [acc_test,acc_train]

def getTrainingValidationError(kvalue, X_test, y_test, X_train, y_train, X_val, y_val):

    print("-----------------" + " K = " +str(kvalue)+ " ---------------------")

    knn = neighbors.KNeighborsClassifier(n_neighbors=kvalue, p=2, metric='euclidean',  n_jobs=-1)

    # Do Training
    t0 = time.time()
    knn.fit(X_train, y_train)
    print("Training done in %0.3fs" % (time.time() - t0) + "\n")

    #////////////////////ACCURACY ON TEST SET
    print("////Test set")
    t0 = time.time()
    predicted = knn.predict(X_test)
    print("Test set done in %0.3fs" % (time.time() - t0))
    acc_test = metrics.accuracy_score(y_test, predicted, normalize=True)
    print("Accuracy score : " + str(metrics.accuracy_score(y_test, predicted, normalize=True)) + "\n")

    #////////////////////ACCURACY ON TRAINING SET
    print("////Training set")
    t0 = time.time()
    predicted2 = knn.predict(X_train)
    print("Training set done in %0.3fs" % (time.time() - t0))
    acc_train = metrics.accuracy_score(y_train, predicted2, normalize=True)
    print("Accuracy score : " + str(metrics.accuracy_score(y_train, predicted2, normalize=True)) + "\n")

    #////////////////////ACCURACY ON VALIDATION SET
    print("////Validation set")
    t0 = time.time()
    predicted3 = knn.predict(X_val)
    print("Validation set done in %0.3fs" % (time.time() - t0))
    acc_val = metrics.accuracy_score(y_val, predicted3, normalize=True)
    print("Accuracy score : " + str(metrics.accuracy_score(y_val, predicted3, normalize=True)) + "\n")

    #[[0.91800000000000004, 1.0, 0.92174999999999996], [0.91308333333333336, 0.96272222222222226, 0.91333333333333333], [0.92500000000000004, 0.96136111111111111, 0.92691666666666672], [0.92549999999999999, 0.95372222222222225, 0.92608333333333337], [0.9291666666666667, 0.95138888888888884, 0.92966666666666664], [0.92666666666666664, 0.94808333333333328, 0.92700000000000005], [0.92608333333333337, 0.94611111111111112, 0.92941666666666667], [0.92474999999999996, 0.94405555555555554, 0.92800000000000005], [0.92483333333333329, 0.94216666666666671, 0.92633333333333334]]
    return [acc_test,acc_train,acc_val]

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
    classifier_group.add_argument('--logistic-regression', action='store_true')
    classifier_group.add_argument('--features-only', action='store_true',
                                  help='only extract features, do not train classifiers')
    parser.add_argument('--learning-curve', action='store_true')
    parser.add_argument('--testing-curve', action='store_true')
    args = parser.parse_args()
    #endregion

    if args.load_features:
        all = []
        # read features from to_pickle
        with (open(args.load_features, "rb")) as openfile:
                try:
                    all.append(pickle.load(openfile))
                except EOFError:print("error")

        #loaded_features = []
        #all_df_temp = []

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

    if args.nearest_neighbors:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

        # create KNN classifier with args.nearest_neighbors as a parameter
        logger.info('Use kNN classifier with k = {}'.format(args.nearest_neighbors))
        knn = neighbors.KNeighborsClassifier(args.nearest_neighbors, n_jobs=-1)

        if args.learning_curve:
            f, ax = plt.subplots(1)

            for kk in [1,2,5]:
                logger.info('Learning curve for KNN with k = ' + str(kk))

                percentage = list(range(0, 100, 10))
                percentage[0] = 1
                num_samples = []
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

                for p in percentage:
                    num_samples.append(math.floor(len(X_train) / 100 * p))
                accuracy_samples_test = []
                accuracy_samples_train = []

                for i in range(0, len(num_samples)):
                    print(str(num_samples[i]) + ' samples')

                    X_train_temp = X_train[:num_samples[i]]
                    y_train_temp = y_train[:num_samples[i]]

                    knn = neighbors.KNeighborsClassifier(kk, n_jobs=-1)

                    res = getTrainingError(kk, X_test, y_test, X_train_temp, y_train_temp, False)

                    accuracy_samples_test.append(res[0])
                    accuracy_samples_train.append(res[1])

                ax.plot(num_samples, accuracy_samples_train,label=str('Train k = '+ str(kk)))
                ax.plot(num_samples, accuracy_samples_test, label=str('Test k = '+ str(kk)))
            plt.ylabel('Accuracy')
            plt.xlabel('Train set size')
            plt.title('KNN with k=1,2,5 (5 is best K)')
            plt.legend(ncol=3, mode="expand", borderaxespad=0.)
            plt.legend(loc="best")
            plt.show()

        if args.testing_curve:
            percentage = list(range(0, 100, 10))
            percentage[0] = 1
            X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
            num_samples = []

            for p in percentage:
                num_samples.append(len(X_test)*(p/100))
            meanaccuracies = []
            stdaccuracies = []
            accuracies = []

            for kk in [1,2,5]:
                print("-------------------- " + "| K = " + str(kk) + "| ---------------------")
                for i in range(0, len(num_samples)):
                    print("Testing 10 times test set with " + str(percentage[i]) + " % samples from test set")
                    for y in range(0,10):

                        X_nothing, X_test_temp, y_nothing, y_test_temp = train_test_split(X_test, y_test, test_size=percentage[i]/100)

                        knn = neighbors.KNeighborsClassifier(kk, n_jobs=-1)

                        # we create an instance of Neighbours Classifier and fit the data.
                        knn.fit(X_train, y_train)

                        result2 = knn.predict(X_test_temp)

                        accuracies.append(metrics.accuracy_score(y_test_temp, result2, normalize=True))
                    meanaccuracies.append(np.mean(accuracies))
                    stdaccuracies.append(np.std(accuracies))
                    print("Mean accuracy : " + str(np.mean(accuracies)))
                    print("Standard deviation : " + str(np.std(accuracies)) + str("\n"))
                    accuracies=[]

                plt.errorbar(num_samples, meanaccuracies, yerr=stdaccuracies, fmt='o', ecolor='r')
                plt.ylabel('Accuracy')
                plt.xlabel('Train set size')
                plt.title('Knn testing curve, k = ' + str(kk))
                plt.show()
                meanaccuracies=[]
                stdaccuracies=[]

        if not args.testing_curve and not args.learning_curve:
        #TRY WITH K=1
            getTrainingError(args.nearest_neighbors, X_test, y_test, X_train, y_train, True)

            getOptimalK2(X,Y)

    elif args.logistic_regression:
        # create KNN classifier with args.nearest_neighbors as a parameter
        logger.info('Use logistic regression')

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


        logreg = linear_model.LogisticRegression(n_jobs=-1)

        # we create an instance of Neighbours Classifier and fit the data.
        logreg.fit(X_train, y_train)

        result = logreg.predict(X_test)
        print(metrics.classification_report(y_test, result))

        print("Accuracy score : " + str(metrics.accuracy_score(y_test, result, normalize=True)))

        if args.learning_curve:

            percentage = list(range(0, 100, 10))
            percentage[0] = 1
            num_samples = []
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

            for p in percentage:
                num_samples.append(math.floor(len(X_train) / 100 * p))
            accuracy_samples_test = []
            accuracy_samples_train = []

            for i in range(0, len(num_samples)):
                logger.info('Testing for ' + str(num_samples[i]) + ' samples')

                X_train_temp = X_train[:num_samples[i]]
                y_train_temp = y_train[:num_samples[i]]

                logreg = linear_model.LogisticRegression(n_jobs=-1)

                # we create an instance of Neighbours Classifier and fit the data.
                logreg.fit(X_train_temp, y_train_temp)

                result = logreg.predict(X_test)
                result2 = logreg.predict(X_train_temp)

                accuracy_samples_test.append(metrics.accuracy_score(y_test, result, normalize=True))
                accuracy_samples_train.append(metrics.accuracy_score(y_train_temp, result2, normalize=True))

            plt.plot(num_samples, accuracy_samples_train)
            plt.plot(num_samples, accuracy_samples_test)
            plt.ylabel('Accuracy')
            plt.xlabel('Train set size')
            plt.show()

        if args.testing_curve:
            percentage = list(range(0, 100, 10))
            percentage[0] = 1
            num_samples = []
            X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

            for p in percentage:
                num_samples.append(p/100)

            meanaccuracies = []
            stdaccuracies = []
            accuracies = []

            for i in range(0, len(num_samples)):
                print("testing 10 times test set with " + str(percentage[i]) + " % samples from test set")
                for y in range(0,10):

                    X_nothing, X_test_temp, y_nothing, y_test_temp = train_test_split(X_test, y_test, test_size=num_samples[i])

                    logreg = linear_model.LogisticRegression()

                    # we create an instance of Neighbours Classifier and fit the data.
                    logreg.fit(X_train, y_train)

                    result2 = logreg.predict(X_test_temp)

                    accuracies.append(metrics.accuracy_score(y_test_temp, result2, normalize=True))
                meanaccuracies.append(np.mean(accuracies))
                stdaccuracies.append(np.std(accuracies))
                print("Mean accuracy : " + str(np.mean(accuracies)))
                print("Standard deviation : " + str(np.std(accuracies)) + str("\n"))
                accuracies=[]


            plt.errorbar(num_samples, meanaccuracies, yerr=stdaccuracies, fmt='o', ecolor='r')
            plt.ylabel('Accuracy')
            plt.xlabel('Train set size')
            plt.title('Logistic regression testing curve')
            plt.show()

    else:
        logger.error('No classifier specified')
        sys.exit()
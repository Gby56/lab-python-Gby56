# Answers for TP3

# K-Nearest Neighbors algorithm (KNN)

## Q5
##### - Test the k-NN with k= 1 on both the training and the test set. Print the score produced by metrics.accuracy_score

```
-------------------- | K = 1| ---------------------
2017-12-23 17:52:01,235 - INFO - Use kNN classifier with k = 1
2017-12-23 17:52:07,919 - INFO - Training done in 6.684s
2017-12-23 17:52:07,919 - INFO - Testing Classifier on test & training sets
Report for prediction on test set
2017-12-23 17:52:21,804 - INFO - Testing  done on test set in 13.884s
             precision    recall  f1-score   support

          0       0.96      0.98      0.97      1221
          1       0.94      0.97      0.95      1342
          2       0.96      0.94      0.95      1148
          3       0.94      0.89      0.91      1234
          4       0.93      0.90      0.91      1120
          5       0.90      0.91      0.91      1069
          6       0.96      0.98      0.97      1204
          7       0.89      0.91      0.90      1197
          8       0.91      0.88      0.89      1228
          9       0.85      0.88      0.86      1237

avg / total       0.92      0.92      0.92     12000

Accuracy score : 0.923916666667
```
```
2017-12-23 17:52:49,688 - INFO - Testing  done on training set in 27.872s
Report for prediction on training set
             precision    recall  f1-score   support

          0       1.00      1.00      1.00      4702
          1       1.00      1.00      1.00      5400
          2       1.00      1.00      1.00      4810
          3       1.00      1.00      1.00      4897
          4       1.00      1.00      1.00      4722
          5       1.00      1.00      1.00      4352
          6       1.00      1.00      1.00      4714
          7       1.00      1.00      1.00      5068
          8       1.00      1.00      1.00      4623
          9       1.00      1.00      1.00      4712

avg / total       1.00      1.00      1.00     48000

Accuracy score : 1.0
```

## Q6
##### Create three sets : train set (60%), validation set (20%) and test set (20%), using twice train_test_split
##### Train a kNN classifier with different values of k and report the train/valid/test accuracy.
##### Select the is best value for k according to the accuracy on the dev set. Report the performance performance of the classifier on the test set for this value of k.
```
COMPUTING OPTIMAL K VALUE, FROM 1 TO 10
----------------- K = 1 ---------------------
Training done in 3.438s

////Test set
Test set done in 11.983s
Accuracy score : 0.920166666667

////Training set
Training set done in 17.797s
Accuracy score : 1.0

////Validation set
Validation set done in 12.480s
Accuracy score : 0.920666666667

----------------- K = 3 ---------------------
Training done in 3.486s

////Test set
Test set done in 15.601s
Accuracy score : 0.92675

////Training set
Training set done in 42.079s
Accuracy score : 0.958833333333

////Validation set
Validation set done in 13.381s
Accuracy score : 0.929166666667

----------------- K = 5 ---------------------
Training done in 3.768s

////Test set
Test set done in 14.597s
Accuracy score : 0.928416666667

////Training set
Training set done in 42.763s
Accuracy score : 0.950222222222

////Validation set
Validation set done in 13.886s
Accuracy score : 0.932833333333

----------------- K = 7 ---------------------
Training done in 3.561s

////Test set
Test set done in 14.586s
Accuracy score : 0.926666666667

////Training set
Training set done in 44.365s
Accuracy score : 0.944166666667

////Validation set
Validation set done in 14.989s
Accuracy score : 0.930333333333

----------------- K = 9 ---------------------
Training done in 3.268s

////Test set
Test set done in 14.890s
Accuracy score : 0.926833333333

////Training set
Training set done in 44.365s
Accuracy score : 0.939555555556

////Validation set
Validation set done in 14.687s
Accuracy score : 0.930416666667

[[0.92016666666666669, 1.0, 0.92066666666666663], [0.92674999999999996, 0.95883333333333332, 0.9291666666666667], [0.92841666666666667, 0.95022222222222219, 0.93283333333333329], [0.92666666666666664, 0.94416666666666671, 0.93033333333333335], [0.92683333333333329, 0.93955555555555559, 0.93041666666666667]]
K = 5 seems to be the best value because of the high accuracy score & minimum error between the predictions on the test and validation set
```
![plot_best_k_train_test_val](https://user-images.githubusercontent.com/6706472/34321300-ed6f2712-e80b-11e7-9d87-26cef228f6bb.png)

# Logistic regression
## Q7

##### Train a Logistic Regression classifier on 80% and test on 20% of the samples. Report the accuracy and compare the the best result of the kNN classifier.
```
             precision    recall  f1-score   support

          0       0.91      0.95      0.93      1169
          1       0.88      0.95      0.91      1331
          2       0.89      0.83      0.86      1230
          3       0.86      0.78      0.82      1297
          4       0.87      0.87      0.87      1130
          5       0.81      0.78      0.79      1103
          6       0.92      0.94      0.93      1192
          7       0.85      0.87      0.86      1234
          8       0.75      0.79      0.77      1141
          9       0.81      0.79      0.80      1173

avg / total       0.86      0.86      0.86     12000

Accuracy score : 0.855833333333
```

## Q8
##### Report the training and test set accuracies for the 1NN, 2NN, kNN (k being the best value for k you previously found) and the Logisitic Regresstion.
##### Plot the training curves on a plot similar to :
![logisticcurve](https://user-images.githubusercontent.com/6706472/34325518-0d165c56-e894-11e7-83fb-af7b9eddfea2.png)

![knncurve](https://user-images.githubusercontent.com/6706472/34325524-4e43326c-e894-11e7-8570-67e815a8b88b.png)

## Q10

##### Report the mean and standard deviation (use np.mean and np.std) of the test set accuracy for the 1NN, 2NN, kNN (k being the best value for k you previously found) and the Logisitic Regresstion.
Knn
```
-------------------- | K = 1| ---------------------
Testing 10 times test set with 1 % samples from test set
2017-12-27 15:22:19,646 - INFO - Use kNN classifier with k = 1
/home/gby/PycharmProjects/lab-python-Gby56/ENV/lib/python3.5/site-packages/sklearn/model_selection/_split.py:2026: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
Mean accuracy : 0.866666666667
Standard deviation : 0.163299316186

Testing 10 times test set with 10 % samples from test set
Mean accuracy : 0.876923076923
Standard deviation : 0.0615384615385

Testing 10 times test set with 20 % samples from test set
Mean accuracy : 0.817307692308
Standard deviation : 0.0432157789505

Testing 10 times test set with 30 % samples from test set
Mean accuracy : 0.838461538462
Standard deviation : 0.0276162297802

Testing 10 times test set with 40 % samples from test set
Mean accuracy : 0.8375
Standard deviation : 0.0305430387967

Testing 10 times test set with 50 % samples from test set
Mean accuracy : 0.850769230769
Standard deviation : 0.0246153846154

Testing 10 times test set with 60 % samples from test set
Mean accuracy : 0.821153846154
Standard deviation : 0.0165801502005

Testing 10 times test set with 70 % samples from test set
Mean accuracy : 0.829120879121
Standard deviation : 0.00996613030067

Testing 10 times test set with 80 % samples from test set
Mean accuracy : 0.831730769231
Standard deviation : 0.00775217091183

Testing 10 times test set with 90 % samples from test set
Mean accuracy : 0.82735042735
Standard deviation : 0.00512820512821

-------------------- | K = 2| ---------------------
Testing 10 times test set with 1 % samples from test set
Mean accuracy : 0.8
Standard deviation : 0.163299316186

Testing 10 times test set with 10 % samples from test set
Mean accuracy : 0.819230769231
Standard deviation : 0.0621365170054

Testing 10 times test set with 20 % samples from test set
Mean accuracy : 0.830769230769
Standard deviation : 0.0453454850867

Testing 10 times test set with 30 % samples from test set
Mean accuracy : 0.815384615385
Standard deviation : 0.0381181249931

Testing 10 times test set with 40 % samples from test set
Mean accuracy : 0.829807692308
Standard deviation : 0.0243442094253

Testing 10 times test set with 50 % samples from test set
Mean accuracy : 0.833846153846
Standard deviation : 0.0182682185954

Testing 10 times test set with 60 % samples from test set
Mean accuracy : 0.828205128205
Standard deviation : 0.0185344003779

Testing 10 times test set with 70 % samples from test set
Mean accuracy : 0.83021978022
Standard deviation : 0.0123961694205

Testing 10 times test set with 80 % samples from test set
Mean accuracy : 0.825480769231
Standard deviation : 0.0136067035559

Testing 10 times test set with 90 % samples from test set
Mean accuracy : 0.82735042735
Standard deviation : 0.00815332650784

-------------------- | K = 5| ---------------------
Testing 10 times test set with 1 % samples from test set
Mean accuracy : 0.766666666667
Standard deviation : 0.152752523165

Testing 10 times test set with 10 % samples from test set
Mean accuracy : 0.819230769231
Standard deviation : 0.0894592565355

Testing 10 times test set with 20 % samples from test set
Mean accuracy : 0.815384615385
Standard deviation : 0.0300394218304

Testing 10 times test set with 30 % samples from test set
Mean accuracy : 0.852564102564
Standard deviation : 0.0304740110872

Testing 10 times test set with 40 % samples from test set
Mean accuracy : 0.832692307692
Standard deviation : 0.0202608726016

Testing 10 times test set with 50 % samples from test set
Mean accuracy : 0.835384615385
Standard deviation : 0.0250915483543

Testing 10 times test set with 60 % samples from test set
Mean accuracy : 0.836538461538
Standard deviation : 0.012246777676

Testing 10 times test set with 70 % samples from test set
Mean accuracy : 0.836813186813
Standard deviation : 0.0173838373841

Testing 10 times test set with 80 % samples from test set
Mean accuracy : 0.839423076923
Standard deviation : 0.0122005553274

Testing 10 times test set with 90 % samples from test set
Mean accuracy : 0.836752136752
Standard deviation : 0.00418716195348
```

Logistic regression :
```
testing 10 times test set with 1 % samples from test set
Mean accuracy : 0.833333333333
Standard deviation : 0.22360679775

testing 10 times test set with 10 % samples from test set
Mean accuracy : 0.796153846154
Standard deviation : 0.0730769230769

testing 10 times test set with 20 % samples from test set
Mean accuracy : 0.798076923077
Standard deviation : 0.0517804308378

testing 10 times test set with 30 % samples from test set
Mean accuracy : 0.814102564103
Standard deviation : 0.0418388945438

testing 10 times test set with 40 % samples from test set
Mean accuracy : 0.8
Standard deviation : 0.0260858845505

testing 10 times test set with 50 % samples from test set
Mean accuracy : 0.821538461538
Standard deviation : 0.027261607918

testing 10 times test set with 60 % samples from test set
Mean accuracy : 0.802564102564
Standard deviation : 0.0268619574869

testing 10 times test set with 70 % samples from test set
Mean accuracy : 0.802747252747
Standard deviation : 0.0144224228004

testing 10 times test set with 80 % samples from test set
Mean accuracy : 0.804326923077
Standard deviation : 0.00913461538462

testing 10 times test set with 90 % samples from test set
Mean accuracy : 0.805128205128
Standard deviation : 0.00667542707342
```

##### Plot the testing curves (mean accuracy) on a plot with error bars (standard deviation of the accuracy), for example using the pandas plot function
Knn
![testcurveknn1](https://user-images.githubusercontent.com/6706472/34384285-8c8e4e7e-eb1b-11e7-85d0-b4ca1d290827.png)

![testcurveknn2](https://user-images.githubusercontent.com/6706472/34384286-8ca8a558-eb1b-11e7-89d5-372024709029.png)

![testcurveknn5](https://user-images.githubusercontent.com/6706472/34384287-8ccbb7c8-eb1b-11e7-8845-0e8ed63ba547.png)
Logistic regression
![testcurvelog](https://user-images.githubusercontent.com/6706472/34384288-8ce76004-eb1b-11e7-98d8-99845b988a38.png)

# Support Vector Machines

## Q9
##### Train SVM with a linear kernel on MNIST with the features previously computed (8x8 subresolution). Use svm.SVC with the option kernel='linear' Report the train and test error rate

```
TEST set : 
             precision    recall  f1-score   support

          0       0.94      0.95      0.95      1128
          1       0.93      0.96      0.94      1346
          2       0.90      0.92      0.91      1192
          3       0.86      0.87      0.87      1221
          4       0.89      0.91      0.90      1200
          5       0.86      0.83      0.85      1082
          6       0.94      0.95      0.95      1163
          7       0.87      0.88      0.88      1284
          8       0.87      0.84      0.85      1154
          9       0.87      0.81      0.84      1230

avg / total       0.89      0.89      0.89     12000

None
Accuracy : 0.893333333333
TRAIN set : 
             precision    recall  f1-score   support

          0       0.95      0.96      0.96      4795
          1       0.93      0.97      0.95      5396
          2       0.91      0.91      0.91      4766
          3       0.87      0.87      0.87      4910
          4       0.90      0.93      0.91      4642
          5       0.86      0.86      0.86      4339
          6       0.95      0.96      0.96      4755
          7       0.89      0.89      0.89      4981
          8       0.87      0.82      0.84      4697
          9       0.87      0.83      0.85      4719

avg / total       0.90      0.90      0.90     48000

None
Accuracy : 0.901416666667
```
##### Train a SVM classifier with a RBF kernel on MNIST with the features previously computed (8x8 subresolution). Use svm.SVC with the option kernel='RBF' with default values for C and gamma. Report its accuracy on the test set.

```
RBF Kernel 

TEST set : 
             precision    recall  f1-score   support

          0       0.95      0.95      0.95        40
          1       0.85      0.94      0.89        49
          2       0.78      0.93      0.85        42
          3       0.89      0.74      0.81        46
          4       0.87      0.89      0.88        37
          5       0.86      0.86      0.86        37
          6       0.97      0.82      0.89        38
          7       0.82      0.84      0.83        37
          8       0.81      0.81      0.81        36
          9       0.81      0.79      0.80        38

avg / total       0.86      0.86      0.86       400

None
Accuracy : 0.8575
```
## Q11
The gamma and C parameters must be optimized for the RBF kernel. Use GridSearchCV to find the best parameters for C in [0.5,1,5] and gamma in [0.05,0.1,0.5]. Report the accuracy of the best model on the test set.
```
RBF Kernel 

Fitting 3 folds for each of 6 candidates, totalling 18 fits
[Parallel(n_jobs=-1)]: Done  18 out of  18 | elapsed:    1.4s finished
TEST set : 
             precision    recall  f1-score   support

          0       0.95      0.95      0.95        44
          1       0.96      0.94      0.95        50
          2       0.75      0.92      0.83        36
          3       0.95      0.85      0.90        41
          4       0.95      0.93      0.94        40
          5       0.90      0.90      0.90        21
          6       0.93      0.93      0.93        43
          7       0.87      0.87      0.87        39
          8       0.91      0.86      0.89        37
          9       0.90      0.90      0.90        49

avg / total       0.91      0.91      0.91       400

None
Accuracy : 0.9075
Best params : {'C': 10, 'gamma': 0.05} using 3 folds with a best score of 0.848125
```

# Q12 BONUS
Download the USPS database and test your best classifier trained on MNIST on the USPS test set. Report the accuracy. Propose one or more solutions to improve this result and test them.
```
2017-12-27 18:53:34,469 - INFO - Loaded feat.pkl file
2017-12-27 18:53:34,492 - INFO - Loaded usps.pkl file
RBF Kernel training on MNIST and testing on USPS

Predicting USPS features to classes...
USPS set : 
             precision    recall  f1-score   support

          0       0.79      0.27      0.40      1194
          1       0.92      0.19      0.32      1005
          2       0.23      0.94      0.37       731
          3       0.26      0.17      0.20       658
          4       0.36      0.42      0.39       652
          5       0.27      0.12      0.17       556
          6       0.59      0.06      0.12       664
          7       0.24      0.44      0.31       645
          8       0.18      0.28      0.22       542
          9       0.08      0.01      0.02       644

avg / total       0.45      0.29      0.27      7291

None
Accuracy : 0.294061171307
on the second run : 0.29392401591
```


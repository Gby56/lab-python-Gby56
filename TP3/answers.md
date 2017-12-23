# Answers for TP3

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

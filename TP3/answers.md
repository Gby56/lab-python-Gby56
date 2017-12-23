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
##### - Select the is best value for k according to the accuracy on the dev set. Report the performance performance of the classifier on the test set for this value of k.

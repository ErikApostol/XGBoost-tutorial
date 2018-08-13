# XGBoost-tutorial

## What is Gradient Boosting
Gradient Boosting is a technique to add weaker algorithms together using a procedure similar to gradient descent, in order to minimize the loss function and make them into a stronger algorithm. New weaker algorithms are added to reduce residual errors of previous algorithms.

## What is XGBoost
XGBoost stands for eXtreme Gradient Boosting, which makes best use of hardware that enables parallel/distributed computing and the use of big datasets, contrary to plain gradient boosting in which each algorithm/tree must be added sequentially. XGBoost is one of the all-round and default algorithm recommended by many Kaggle winners.
## Use XGBoost with Python and sklearn
* Use XGBClassifier for classification and XGBRegressor for regression. 
* from sklearn.metrics import accuracy_score
```Python
accuracy = accuracy_score(label, prediction)
```
## Early stopping
You can demand XGBoost to monitor the test/dev set performance, and to stop after the performance does not improve for a number of rounds:
```Python
model.fit(X_train, Y_train, early_stopping_rounds=10, eval_metric="error", eval_set=[(X_test, Y_test)], verbose=True)
```

## Feature importance
To print the importance of features, use the feature_importances_ attribute of the trained model:
```Python
print(model.feature_importances_)
```

You can also plot the features ordered by their importances:
```Python
from xgboost import plot_importance
from matplotlib import pyplot
plot_importance(model)
pyplot.show()
```

## Parameters
* `learning_rate`: less than 0.1 is recommended, smaller value helps.
* `n_estimators`: number of boosted trees to fit. 
* `max_depth`: maximum depth of a tree. 2-to-8 is recommended. Deeper trees do not benefit. Decreasing it helps reducing overfitting.
* `subsample`: subsample ratio of training instances. 30%-80% is recommended.
### Parameter tuning strategy 1
1. Use default parameter values and plot learning curves of training and evaluation sets?
1. If overfitting, 
    * decrease learning rate 
    * increase number of trees
1. If underfitting,
    * increase learning rate 
    * decrease number of trees
### Parameter tuning strategy 2
(Owen Zhang, Kaggle competition winner) Set number of trees to 100 or 1000, and tune the learning rate.
difference of fit and train in xgboost

For an example of grid-searching parameters, see [tuning.py](https://github.com/ErikApostol/XGBoost-tutorial/blob/master/tuning.py).

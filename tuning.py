from pandas import read_csv
from numpy import loadtxt, linspace
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# .loc means selecting by label (not index) and conditional statement, can selecty by row and column
# .iloc means integer location, can only select by row
dataset = read_csv("pima-indians-diabetes.data.csv", header=None)
X = dataset.loc[:, 0:7]
Y = dataset.loc[:,8]
print(X.shape)
print(Y.shape)
#dataset = loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
#X = dataset[:,0:8]
#Y = dataset[:,8]

model = XGBClassifier()

n_estimators = [50, 100, 200, 1000]
learning_rate = 10**(linspace(-1, -4, 10))
max_depth = linspace(2, 8, 4).astype(int)
param_grid = dict(n_estimators = n_estimators, learning_rate = learning_rate, max_depth = max_depth)
grid_search = GridSearchCV(model, param_grid, scoring="accuracy", n_jobs=-1, cv=10)
# n_jobs=-1 means that the computation will be dispatched on all the CPUs of the computer.
grid_result = grid_search.fit(X, Y)

print("Best score %f using parameters %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("Mean score %f and deviation %f with parameters %r" % (mean, stdev, param))

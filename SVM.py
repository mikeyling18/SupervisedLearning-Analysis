import matplotlib.pylab as pl
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.interpolate import spline
from sklearn.svm import SVC


def create_graphs_svm(X, Y, kernels, dataset_name, gamma, exponent, max_iter):
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.25)
    smooth = np.linspace(max_iter.min(), max_iter.max(), 300)
    all_test_acc = list()
    all_train_acc = list()
    all_time_train = list()
    all_time_predict = list()
    # Get accuracy for each value of gamma

    # for g in gamma:
    for e in exponent:
        one_test_acc_list = list()
        one_train_acc_list = list()
        one_time_train_list = list()
        one_time_predict_list = list()

        for M in max_iter:
            # clf = SVC(kernel='rbf', gamma=g, max_iter=M)
            clf = SVC(kernel = 'poly', degree=e, max_iter=M)
            # Train the model
            start_time = time.time()
            clf.fit(train_X, train_Y)
            train_time = time.time() - start_time

            # Get Predictions
            start_time = time.time()
            original_predicted_test_y = clf.predict(test_X)
            predict_time = time.time() - start_time

            original_predicted_train_y = clf.predict(train_X)

            # Get Accuracies
            train_acc = accuracy_score(train_Y, original_predicted_train_y)
            test_acc = accuracy_score(test_Y, original_predicted_test_y)

            one_train_acc_list.append(train_acc)
            one_test_acc_list.append(test_acc)
            one_time_predict_list.append(predict_time)
            one_time_train_list.append(train_time)

        all_train_acc.append(one_train_acc_list)
        all_test_acc.append(one_test_acc_list)
        all_time_train.append(one_time_train_list)
        all_time_predict.append(one_time_predict_list)

    plot_names = list()
    for name in exponent:
        name = 'Poly Degree: %s' % name
        plot_names.append(name)
    #
    colors = pl.cm.tab20(np.linspace(0, 1, 19))
    for i in range(len(exponent)):
        # pl.plot(smooth, 100 * spline(max_iter, all_train_acc[i], smooth), color=colors[2 * (i - 1)])
        pl.plot(smooth, 100 * spline(max_iter, all_test_acc[i], smooth), color=colors[2 * (i - 1)])
    pl.xlabel("Iterations")
    pl.ylabel("Accuracy (%)")

    pl.legend(plot_names, loc='upper left', bbox_to_anchor=(1.05, 1))
    pl.title('%s - SVM (Poly Kernel) Accuracy vs Iterations with Varying Degrees' % dataset_name)
    pl.show()


    for i in range(len(all_time_train)):
        pl.plot(smooth, spline(max_iter, all_time_train[i], smooth), color=colors[2 * (i - 1)])
    pl.xlabel("Iterations")
    pl.ylabel("Time Taken (sec)")

    pl.legend(plot_names, loc='upper left', bbox_to_anchor=(1.05, 1))
    pl.title('%s - SVM (Poly Kernel) Training Time vs Iterations with Varying Degrees' % dataset_name)
    pl.show()

    for i in range(len(all_time_predict)):
        pl.plot(smooth, spline(max_iter, all_time_predict[i], smooth), color=colors[2 * (i - 1)])
    pl.xlabel("Iterations")
    pl.ylabel("Time Taken (sec)")

    pl.legend(plot_names, loc='upper left', bbox_to_anchor=(1.05, 1))
    pl.title('%s - SVM (Poly Kernel) Predicting Time vs Iterations with Varying Degrees' % dataset_name)
    pl.show()



kernels = ['rbf', 'poly']
gamma = [0.01, 0.05, 0.1]
exponent = [1, 2, 3, 4, 5]
max_iterations = np.linspace(100, 5000, 10)
# =============================================================
# Adult Dataset
# =============================================================
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race',
                'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country', '50K']

data = pd.read_csv("adult.csv", names = column_names, index_col = False)
del data['relationship']
del data['fnlwgt']
del data['education']

new_column_names = list(data)
data = data[~data[new_column_names].isin([' ?'])]
data = data.dropna()

object_cols = data.loc[:, [data.dtypes == object][0].values]
object_cols_names = list(object_cols)
data[object_cols_names] = data[object_cols_names].astype('category')
for name in object_cols_names:
    data[name] = data[name].cat.codes

data['education-num'] = data['education-num'].astype('int')

Y = data['50K'].values
del data['50K']
X = data.values

# create_graphs_svm(X, Y, kernels, "Adult Dataset", gamma, exponent, max_iterations)


# =============================================================
# Abalone Dataset
# =============================================================

column_names = ['sex', 'length', 'diameter', 'height', 'whole weight', 'shucked weight', 'viscera weight',
                'shell weight', 'rings']

data = pd.read_csv("abalone.csv", names = column_names, index_col = False)

# Create Labels
Y = data["sex"].values

# Drop Labels from dataframe
del data["sex"]

# Create Samples
X = data.values.astype(np.float)

create_graphs_svm(X, Y, kernels, "Abalone Dataset", gamma, exponent, max_iterations)
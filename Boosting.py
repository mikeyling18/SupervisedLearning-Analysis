import matplotlib.pylab as pl
import numpy as np
import pandas as pd
import time

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.interpolate import spline
from sklearn.ensemble import AdaBoostClassifier


def create_graphs(train_size, minimum_leaves, X, Y, name):
    train_acc_list = list()
    test_acc_list = list()

    predict_time_list = list()
    train_time_list = list()

    smooth = np.linspace(train_size.min(), train_size.max(), 300)

    for min_leaf in minimum_leaves:
        train_accuracies = list()
        test_accuracies = list()
        train_times = list()
        predict_times = list()
        for size in training_sizes:
            # original_tree = tree.DecisionTreeClassifier(min_samples_leaf=min_leaf)
            boosted_tree = AdaBoostClassifier(tree.DecisionTreeClassifier(min_samples_leaf=min_leaf))
            # Split samples and labels
            train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=(1.0 - size))

            # Train the model
            start_time = time.time()
            boosted_tree.fit(train_X, train_Y)
            train_time = time.time() - start_time

            # Get Predictions
            start_time = time.time()
            original_predicted_test_y = boosted_tree.predict(test_X)
            predict_time = time.time() - start_time

            original_predicted_train_y = boosted_tree.predict(train_X)

            # Get Accuracies
            train_acc = accuracy_score(train_Y, original_predicted_train_y)
            test_acc = accuracy_score(test_Y, original_predicted_test_y)

            # Construct Accuracy Lists
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            # Construct Time Lists
            predict_times.append(predict_time)
            train_times.append(train_time)

        train_acc_list.append(train_accuracies)
        test_acc_list.append(test_accuracies)

        predict_time_list.append(predict_times)
        train_time_list.append(train_times)

    # Plot Everything!
    smooth_perc = smooth * 100
    colors = pl.cm.tab20(np.linspace(0, 1, 19))
    for i in range(len(train_acc_list)):
        pl.plot(smooth_perc, 100 * spline(training_sizes, train_acc_list[i], smooth), color=colors[2 * (i - 1)])
        pl.plot(smooth_perc, 100 * spline(training_sizes, test_acc_list[i], smooth), color=colors[(2 * i) - 1])
    pl.xlabel("Training Size (%)")
    pl.ylabel("Accuracy (%)")

    graph_names = ['Train - Number of Leaves:5', 'Test - Number of Leaves:5',
                   'Train - Number of Leaves:10', 'Test - Number of Leaves:10',
                   'Train - Number of Leaves:25', 'Test - Number of Leaves:25',
                   'Train - Number of Leaves:50', 'Test - Number of Leaves:50',
                   'Train - Number of Leaves:100', 'Test - Number of Leaves:100']
    pl.legend(graph_names, loc='upper left', bbox_to_anchor=(1.05, 1))
    pl.title('%s - Boosted Decision Tree Learning Curve \n with Varying Minimum Leaf Number and Training Size' % name)
    pl.show()

    for i in range(len(train_acc_list)):
        pl.plot(smooth_perc, spline(training_sizes, train_time_list[i], smooth), color=colors[2 * (i - 1)])
    pl.xlabel("Training Size (%)")
    pl.ylabel("Time Taken (sec)")

    train_time_graph_names = ['Number of Leaves: 5',
                              'Number of Leaves: 10',
                              'Number of Leaves: 25',
                              'Number of Leaves: 50',
                              'Number of Leaves: 100']
    pl.legend(train_time_graph_names, loc='upper left', bbox_to_anchor=(1.05, 1))
    pl.title('%s - Training Time vs. Training Size' % name)
    pl.show()

    for i in range(len(train_acc_list)):
        pl.plot(smooth_perc, spline(training_sizes, predict_time_list[i], smooth), color=colors[2 * (i - 1)])
    pl.xlabel("Training Size (%)")
    pl.ylabel("Time Taken (sec)")

    train_time_graph_names = ['Number of Leaves: 5',
                              'Number of Leaves: 10',
                              'Number of Leaves: 25',
                              'Number of Leaves: 50',
                              'Number of Leaves: 100']
    pl.legend(train_time_graph_names, loc='upper left', bbox_to_anchor=(1.05, 1))
    pl.title('%s - Prediction Time vs. Training Size' % name)
    pl.show()


# Decision Tree and Graph Parameters
training_sizes = np.linspace(0.5, 0.95, 20)
smooth = np.linspace(training_sizes.min(), training_sizes.max(), 300)
min_leaf_num = [5, 10, 25, 50, 100]

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

create_graphs(training_sizes, min_leaf_num, X, Y, "Adult Dataset")


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

create_graphs(training_sizes, min_leaf_num, X, Y, "Abalone Dataset")
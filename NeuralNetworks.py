import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import time

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.interpolate import spline
from sklearn.model_selection import KFold


def create_graphs_NN(X, Y, learning_rate, max_iterations, dataset_name, random_state):



    train_accuracies = list()
    test_accuracies = list()
    train_times = list()
    predict_times = list()

    smooth = np.linspace(max_iterations.min(), max_iterations.max(), 300)
    for alpha in learning_rate:


        train_acc_list_iter = list()
        test_acc_list_iter = list()
        time_train_list_iter = list()
        time_predict_list_iter = list()
        for max_iter in max_iterations:

            nn = MLPClassifier(learning_rate_init=alpha, activation='logistic', solver='sgd', max_iter=max_iter)

            # Create Training and Testing Data
            # train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=.20)

            kfold = KFold(n_splits = 5, shuffle=True)

            train_acc_list = list()
            test_acc_list = list()
            time_train_list = list()
            time_predict_list = list()

            for train_index, test_index in kfold.split(X):
                train_X, test_X = X[train_index], X[test_index]
                train_Y, test_Y = Y[train_index], Y[test_index]
                # Create and Train Model

                # Train and Get Time
                time_start = time.time()
                nn.fit(train_X, train_Y)
                time_train = time.time() - time_start

                time_start = time.time()
                predict_Y_test = nn.predict(test_X)
                time_predict = time.time() - time_start

                predict_Y_train = nn.predict(train_X)

                test_acc = accuracy_score(test_Y, predict_Y_test)
                train_acc = accuracy_score(train_Y, predict_Y_train)

                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                time_train_list.append(time_train)
                time_predict_list.append(time_predict)

            train_acc_list_iter.append(np.mean(train_acc_list))
            test_acc_list_iter.append(np.mean(test_acc_list))
            time_train_list_iter.append(np.mean(time_train_list))
            time_predict_list_iter.append(np.mean(time_predict_list))

        train_accuracies.append(train_acc_list_iter)
        test_accuracies.append(test_acc_list_iter)
        predict_times.append(time_predict_list_iter)
        train_times.append(time_train_list_iter)

    colors = plt.cm.tab20(np.linspace(0, 1, 19))
    plot_names = list()
    for name in learning_rate:
        name = 'Learning Rate: %s' % name
        plot_names.append(name)

    plot_names = np.asarray(plot_names)
    for i in range(len(learning_rate)):
        # plt.plot(smooth_perc, 100 * spline(train_sizes, train_acc_list_iter[i], smooth))
        plt.plot(smooth, 100*spline(max_iterations, test_accuracies[i], smooth), color=colors[2 * (i - 1)])
        # plt.plot(max_iterations, test_accuracies[i])
    plt.legend(plot_names, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title('%s - Accuracy vs Epochs with Varying Learning Rates' % dataset_name)
    plt.show()

    for i in range(len(learning_rate)):
        plt.plot(smooth, spline(max_iterations, train_times[i], smooth), color=colors[2 * (i - 1)])
    plt.legend(plot_names, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.xlabel("Epochs")
    plt.ylabel("Time (sec)")
    plt.title('%s - Training Time vs Epochs with Varying Learning Rates' % dataset_name)
    plt.show()

    for i in range(len(learning_rate)):
        plt.plot(smooth, spline(max_iterations, predict_times[i], smooth), color=colors[2 * (i - 1)])
    plt.legend(plot_names, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.xlabel("Epochs")
    plt.ylabel("Time (sec)")
    plt.title('%s - Predicting Time vs Epochs with Varying Learning Rates' % dataset_name)
    plt.show()


# =============================================================
# Adult Dataset
# =============================================================
# Parameters
max_iterations = np.linspace(500, 5000, 10).astype('int')
adult_learning_rates = [0.02, 0.04, 0.06, 0.08, 0.1]
random_state = 10

# Load Data
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race',
                'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country', '50K']

data = pd.read_csv("adult.csv", names = column_names, index_col = False)

# Clean Data
del data['relationship']
del data['fnlwgt']

new_column_names = list(data)
data = data[~data[new_column_names].isin([' ?'])]
data = data.dropna()

col_dtypes = data.dtypes
obj_col_names = col_dtypes.index[col_dtypes.values == 'object'].values

for col in obj_col_names:
    data[col] = data[col].astype('category')
    data[col] = data[col].cat.codes

data = data[0: int(len(data))]
# Create Samples and Labels
Y = data['50K'].values
del data['50K']
X = data.values

# create_graphs_NN(X, Y, adult_learning_rates, max_iterations, 'Adult Dataset', random_state)


# =============================================================
# Abalone Dataset
# =============================================================
abalone_learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
max_iterations = np.linspace(500, 5000, 10).astype('int')

column_names = ['sex', 'length', 'diameter', 'height', 'whole weight', 'shucked weight', 'viscera weight',
                'shell weight', 'rings']

data = pd.read_csv("abalone.csv", names = column_names, index_col = False)

# Create Labels
Y = data["sex"].values

# Drop Labels from dataframe
del data["sex"]

# Create Samples
X = data.values.astype(np.float)

create_graphs_NN(X, Y, abalone_learning_rates, max_iterations, 'Abalone Dataset', random_state)
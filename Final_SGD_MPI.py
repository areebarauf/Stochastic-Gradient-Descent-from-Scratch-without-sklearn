from itertools import chain
import pandas as pd
import numpy as np
from numpy import random
from mpi4py import MPI
from math import sqrt
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def _indexing(x, indices):
    # np array indexing
    if hasattr(x, 'shape'):
        return x[indices]

    # list indexing
    return [x[idx] for idx in indices]


def train_test_split(*arrays, test_size=0.3, shuffle=True, random_seed=1):
    # checks
    assert 0 < test_size < 1
    assert len(arrays) > 0
    length = len(arrays[0])
    for i in arrays:
        assert len(i) == length

    n_test = int(np.ceil(length * test_size))
    n_train = length - n_test

    if shuffle:
        perm = np.random.RandomState(random_seed).permutation(length)
        test_indices = perm[:n_test]
        train_indices = perm[n_test:]
    else:
        train_indices = np.arange(n_train)
        test_indices = np.arange(n_train, length)

    return list(chain.from_iterable((_indexing(x, train_indices), _indexing(x, test_indices)) for x in arrays))


def Stochastic_Gradient_Descent(training_data_, XTest, YTest, learning_rate, epochs, random_sub_set, reducing_rate):
    # selecting initial weight and intercept as zero for the first iteration gradient
    weight = np.zeros(shape=(1, training_data_.shape[1] - 1))
    intercept = 0
    current_epoch = 1
    local_mse = []
    timeList = []

    while current_epoch <= epochs:
        random_temp = training_data_.sample(random_sub_set)
        random_y = np.array(random_temp['target'])
        random_x = np.array(random_temp.drop('target', axis=1))

        weights_gradient = np.zeros(shape=(1, training_data_.shape[1] - 1))
        intercept_gradient = 0

        for itr in range(random_sub_set):
            # linear regression line
            prediction = np.dot(weight, random_x[itr]) + intercept
            # derivative of equation of weight and intercept implemented below
            weights_gradient = weights_gradient + (-2) * random_x[itr] * (random_y[itr] - prediction)
            intercept_gradient = intercept_gradient + (-2) * (random_y[itr] - prediction)

        weight = weight - learning_rate * (weights_gradient / random_sub_set)
        intercept = intercept - learning_rate * (intercept_gradient / random_sub_set)
        current_epoch = current_epoch + 1
        learning_rate = learning_rate / reducing_rate
        y_pred_epoch = local_SGD_predict(XTest, weight, intercept)
        mse_ranks = root_mean_square_error_(YTest, y_pred_epoch)
        local_mse.append(mse_ranks)
        timeList.append(MPI.Wtime())

    return weight, intercept, local_mse, timeList


def root_mean_square_error_(actual, predicted):
    sum_error = 0.0
    for itr in range(len(actual)):
        prediction_error = predicted[itr] - actual[itr]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


def local_SGD_predict(x_train_data, weight_array, intercept):
    y_pred = []
    for itr in range(len(x_train_data)):
        Y_predicted = np.asscalar(np.dot(weight_array, x_train_data[itr]) + intercept)
        y_pred.append(Y_predicted)
    return np.array(y_pred)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0
start_time = MPI.Wtime()
batch_size = 1000

if rank == root:
    # Loading Data from CSV File
    file_data = pd.read_csv('file_data.csv')
    X = file_data.iloc[:, 0:265].to_numpy()
    Y = file_data.iloc[:, -1].to_numpy()
    # Split Data into train and test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)

    # print("X Shape: ", X.shape)
    # print("Y Shape: ", Y.shape)
    # print("X_train Shape: ", x_train.shape)
    # print("Y_train Shape: ", y_train.shape)
    # ---------- standardizing data----------
    norm = np.linalg.norm(x_train)
    normal_array = x_train / norm
    xtest_norm = x_test / np.linalg.norm(x_test)
    # ------------Adding the target Column in the training data---------
    train_data = pd.DataFrame(normal_array)
    train_data['target'] = y_train
    x_test = np.array(xtest_norm)
    y_test = np.array(y_test)
    # split array in the number of workers opened
    split_array_train_data = np.array_split(train_data, size)


else:
    x_test = None
    y_test = None
    split_array_train_data = None

train_data = comm.scatter(split_array_train_data, root=root)
# print(f'rank:{rank} got this size of training Data:{len(train_data)}')
x_test = comm.bcast(x_test, root=root)
y_test = comm.bcast(y_test, root=root)
weight_, intercept_, MSE_gathered, time_gathered_algo = Stochastic_Gradient_Descent(train_data, x_test, y_test,
                                                                                    learning_rate=0.01, epochs=200,
                                                                                    random_sub_set=1000,
                                                                                    reducing_rate=1.1)
times_each_epoc_gather = comm.gather(time_gathered_algo, root=root)
mse_each_epoch_Gather = comm.gather(MSE_gathered, root=root)
print(f'MSE gathered:\n {mse_each_epoch_Gather}')
weights_reduced = comm.reduce(weight_, root=root)
intercepts_reduced = comm.reduce(intercept_, root=root)

end_time = MPI.Wtime()
Net_time = end_time - start_time
all_times = comm.gather(Net_time, root=0)

if rank == 0:
    times = np.vstack(all_times)
    print('times in Array:', Net_time)
    time_sum = np.sum(times)

    # print('Total Time for processes is Net_time=%.3f' % time_sum)
    average_weights = weights_reduced / size
    average_intercept = intercepts_reduced / size
    print(f'rank:{rank} weights are:{average_weights}\n')
    print(f'rank:{rank} coefficient is:{average_intercept}\n')
    y_pred_local = local_SGD_predict(x_test, average_weights, average_intercept)
    print('Mean Squared Error :', root_mean_square_error_(y_test, y_pred_local))
    #
    for i, row in enumerate(mse_each_epoch_Gather):
        plt.plot(range(len(row)), row, label='rank ' + str(i))
    plt.xlabel('Number of Iterations')
    plt.ylabel('RMSE')
    plt.title('RMSE Vs Number of Iterations')
    plt.grid(ls='--')
    plt.legend()
    plt.show()

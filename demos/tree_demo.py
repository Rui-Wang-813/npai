import sys
sys.path.append("..")
sys.path.append("../npai/")

from npai.machine_learning import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def classification_demo():
    #---------------Read the training and testing files
    DATA_DIR = "D:/DATA/npml-tree-demos/clf/"
    training_data=np.genfromtxt(DATA_DIR + 'train_decision_tree.csv', delimiter=',')
    X_train = training_data[:,0:2]
    y_train = training_data[:,2]

    tree = DecisionTreeClassifier(max_depth=10)
    tree.fit(X_train, y_train)

    testing_data=np.genfromtxt(DATA_DIR + 'test_decision_tree.csv', delimiter=',')
    X_test = testing_data[:,0:2]
    y_test = testing_data[:,2]

    preds = tree.transform(X_test)
    acc = ((preds == y_test) / y_test.shape[0]).sum()
    print(f"Accuracy is {acc:.4f}")

    x_space = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]), 500)
    y_space = np.linspace(np.min(X_train[:, 1]), np.max(X_train[:, 1]), 500)
    X = np.array([[x, y] for x in x_space for y in y_space])
    Z_pred = tree.transform(X)
    plt.scatter(X[:, 0], X[:, 1], s=1, c=["orange" if z == 0 else "white" for z in Z_pred])
    for i, x in enumerate(X_test):
        y = y_test[i]
        plt.scatter(x[0], x[1], marker='o' if y == 0 else 'x', c="red" if y == 0 else "black")
    plt.show()

def regression_demo():
    # DATA_DIR = "D:/DATA/Housing/"
    # train_df = pd.read_csv(DATA_DIR + "train.csv")
    # test_df = pd.read_csv(DATA_DIR + "test.csv")
    # train_data, test_data = train_df.values, test_df.values

    # X_train, y_train = train_data[:, 1:], train_data[:, 0]
    # X_test, y_test = test_data[:, 1:], test_data[:, 0]

    X_train = np.linspace(0, 4, 100)
    y_train = np.sin(X_train)
    X_train = X_train.reshape((-1, 1))

    X_test, y_test = X_train, y_train

    tree = DecisionTreeRegressor(max_depth=10, split_rule="mse").fit(X_train, y_train)
    preds = tree.transform(X_test)
    mse = np.sqrt(np.mean((preds - y_test) ** 2))
    print(f"Mean error is {mse:.4f}")

    plt.scatter(X_train, y_train, c="red")
    plt.plot(X_train, preds, linewidth=2)
    plt.show()


if __name__ == "__main__":
    # classification_demo()
    regression_demo()
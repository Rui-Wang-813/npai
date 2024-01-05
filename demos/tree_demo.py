import sys
sys.path.append("..")
sys.path.append("../npai/")

from npai.machine_learning import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

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
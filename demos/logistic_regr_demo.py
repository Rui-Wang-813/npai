import sys
sys.path.append("..")
sys.path.append("../npai/")

from npai.machine_learning import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the clusters
n_samples = 500  
mean1 = [2, 2]  
mean2 = [-2, -2] 
cov = [[1, 0], [0, 1]]  

# Generate two clusters
cluster1 = np.random.multivariate_normal(mean1, cov, n_samples)
cluster2 = np.random.multivariate_normal(mean2, cov, n_samples)

# Labels for the clusters
labels1 = np.zeros(n_samples)  # Class 0
labels2 = np.ones(n_samples)   # Class 1

# Combine the clusters to form the dataset
X = np.vstack((cluster1, cluster2))
y = np.hstack((labels1, labels2))

model = LogisticRegression()
model.fit(X, y, verbose = False)

#pred the data and cal accuracy
preds = model.transform(X)
acc = np.sum(preds == y) / len(y)
print(f"Accuracy: {acc}")

# Coefficients for the decision boundary line
coef = model.w
intercept = model.b

# Calculate the slope and y-intercept of the decision boundary line
slope = -coef[0] / coef[1]
y_intercept = -intercept / coef[1]

# Define the decision boundary line
x_values = np.array([X[:, 0].min(), X[:, 0].max()])
y_values = slope * x_values + y_intercept

# Plotting the dataset and decision boundary line
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.plot(x_values, y_values, label="Decision Boundary", color='green')

plt.title('Logistic Regression - Decision Boundary Line')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()






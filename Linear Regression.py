import numpy as np

# Input number of features
n_features = int(input('Enter number of columns (features): '))

# Input each feature column
X = []
for i in range(n_features):
    col = list(map(float, input(f'Enter values for X{i+1} (space-separated): ').split()))
    X.append(col)

# Input target variable Y
Y = list(map(float, input('Enter values for Y (space-separated): ').split()))
Y = np.array(Y)

# Validate feature and label length
n_samples = len(Y)
for i in range(n_features):
    if len(X[i]) != n_samples:
        print(f"Length mismatch in X{i+1} and Y")
        exit()

# Convert feature matrix X to shape (n_samples, n_features)
X = np.array(X).T  # Transpose to shape (samples, features)

# Bias column 
bias = float(input("Enter Bias value: "))

if bias != 0:
    bias_column = np.ones((n_samples, 1)) * bias
    X_bias = np.hstack((bias_column, X))
else:
    X_bias = X  

# beta = (X^T * X)^-1 * X^T * Y
X_transpose = X_bias.T
XtX = np.dot(X_transpose, X_bias)

# Use pseudoinverse to avoid inversion errors
XtX_inv = np.linalg.pinv(XtX)

XtY = np.dot(X_transpose, Y)
beta = np.dot(XtX_inv, XtY)

# Make predictions
y_pred = np.dot(X_bias, beta)

# Output
if bias != 0:
    print(f"\nBias column value used: {bias}")
    print("Bias/Intercept term (beta[0]) -", round(beta[0], 4))
    start_idx = 1
else:
    print("\nNo bias/intercept used.")
    start_idx = 0

for i in range(start_idx, len(beta)):
    print(f"Slope for X{i if bias == 0 else i} -", round(beta[i], 4))

print("\nPredicted Values:", y_pred)

# R² - Coefficient of Determination
ss_res = np.sum((Y - y_pred) ** 2)
ss_tot = np.sum((Y - np.mean(Y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

accuracy_percent = r_squared * 100
print(f"\nPrediction's Accuracy: {accuracy_percent:.2f}%")
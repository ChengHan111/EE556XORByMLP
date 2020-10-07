# We have the target for XOR
# 1 1 --> 0
# 0 1 --> 1
# 1 0 --> 1
# 0 0 --> 0
import numpy as np
import matplotlib.pyplot as plt
# Activation function:sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# def sigmoid derivative for the backprop
def sigmoid_drivative(x):
    return sigmoid(x)*(1 - sigmoid(x))

# Forward function
def forward(x, w1, w2, predict = False):
    a1 = np.matmul(x, w1)
    z1 = sigmoid(a1)

#     create and add bias
    bias = np.ones((len(z1), 1))
    #  rows joining 4x5 join 4x1 == 4x6
    z1 = np.concatenate((bias, z1),axis=1)
    # print(z1.shape)
    a2 = np.matmul(z1, w2)
    z2 = sigmoid(a2)
    if predict:
        return z2
    return a1,z1,a2,z2

# Backprop function
def backprop(a2,z0,z1,z2,y):
    delta2 = z2 - y
    Delta2 = np.matmul(z1.T, delta2)
    delta1 = (delta2.dot(w2[1:,:].T))*sigmoid_drivative(a1)
    Delta1 = np.matmul(z0.T, delta1)
    return delta2, Delta1, Delta2


# The first column of it is bias
X = np.array([[1,1,0],
             [1,0,1],
             [1,0,0],
             [1,1,1]])
# output
y = np.array([[1],[1],[0],[0]])

# init weights
w1 = np.random.randn(3,5)
w2 = np.random.randn(6,1)

# init learning rate
lr = 0.09

costs = []

# init epochs
epochs = 15000

m = len(X)

# Start training
for i in range(epochs):

    # Forward
    a1,z1,a2,z2 = forward(X,w1,w2)

    # Backprop
    delta2, Delta1, Delta2 = backprop(a2,X,z1,z2,y)

    w1 -= lr*(1/m)*Delta1
    w2 -= lr*(1/m)*Delta2

    # Add costs to list for plotting
    c = np.mean(np.abs(delta2))
    costs.append(c)

    if i % 1000 == 0:
        print(f"Iteration:{i}. Error: {c}")

    # Training complete
print("Training Complete")

    # Make predictions
z3 = forward(X, w1, w2, True)
print("Percentages: ")
print(z3)
print("Predictions: ")
print(np.round(z3))


    # Plot cost

plt.plot(costs)
plt.show()

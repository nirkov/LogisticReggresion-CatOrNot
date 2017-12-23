

import matplotlib.pyplot as plt
import numpy as np
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
NUM_OF_TRAIN_DATA = train_set_y.shape[1];
NUM_OF_TEST_DATA = test_set_y.shape[1];
NUM_OF_PIXELS =  train_set_x_orig.shape[1];

#Flatten the images from tenzor to vector of pixels with numpy.reshape
flattenTrainSet = train_set_x_orig.reshape(train_set_x_orig.shape[0] , -1).T / 255.
flattenTestSet = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T / 255.

print("Flatten train set shape : " + str(flattenTrainSet.shape))
print("Flatten test set shape  : " + str(flattenTestSet.shape))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initializeParameter(dimantion):
    w = np.zeros(shape=(dimantion, 1))
    b = 0
    return w, b

def propogate (w , b , pixelCulMatrix  , trueLabelVector):
    X = pixelCulMatrix
    Y = trueLabelVector
    numOfImage = X.shape[1]
    sigmoidResult = sigmoid(np.dot(w.T, pixelCulMatrix) + b)
    costFunction = - np.sum(Y * np.log(sigmoidResult) + ((1-Y) * np.log(1-sigmoidResult)))/numOfImage
    derivative_w = np.dot(X, (sigmoidResult - Y).T) / numOfImage
    derivative_b = np.sum(sigmoidResult - Y) / numOfImage
    gradient = {"dw": derivative_w, "db": derivative_b}
    return gradient, np.squeeze(costFunction)

def optimization_By_GradientDescent (w, b, X, Y, numIteration, learningRate,  printCost=False):
    costs = list()
    for index in range(numIteration):
        gradient, cost = propogate(w, b, X, Y)
        dw = gradient["dw"]
        db = gradient["db"]
        w = w - learningRate * dw
        b = b - learningRate * db
        if printCost and index % 100 == 0:
            costs.append(cost)
            print("Cost after iteration %i: %f" % (index, cost))

    parameter_after_optimization = {"w": w, "b": b}
    gradient = {"dw":dw,"db":db}
    return parameter_after_optimization, gradient, cost

def predictFunction(w, b , X):
    A = sigmoid(np.dot(w.T , X) + b)
    Y = np.zeros(shape=(1, A.shape[1]))
    for k in range(A.shape[1]):
        if A[0, k] > 0.5: Y[0, k] = int(1)
    assert(Y.shape == A.shape)
    return Y

def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost):
    w, b = initializeParameter(X_train.shape[0])
    optimized_param, gradient, cost = optimization_By_GradientDescent(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    test_prediction_vector = predictFunction(optimized_param["w"], optimized_param["b"], X_test)
    train_prediction_vector = predictFunction(optimized_param["w"], optimized_param["b"], X_train)
    print("Train accuracy : {}%".format(100-np.mean(np.abs(train_prediction_vector - Y_train)*100)))
    print("Test accuracy : {}%".format(100 - np.mean(np.abs(test_prediction_vector - Y_test) * 100)))
    return {"Cost": cost, "Prediction test":test_prediction_vector, "Prediction train":train_prediction_vector
            , "w":optimized_param["w"], "b":optimized_param["b"], "Learning rate":learning_rate, "Number of iteration":num_iterations }



CLASSIFICATION_RESULT = model(flattenTrainSet, train_set_y, flattenTestSet, test_set_y, 2000, 0.005, True)
start = True
while start:
    try:
        index = int(input("Insert number between 0 to 49 : "))
        if index in range(1,51):
            index = index - 1
            plt.imshow(flattenTestSet[:, index].reshape((NUM_OF_PIXELS, NUM_OF_PIXELS, 3)))
            print("y = " + str(test_set_y[0, index]) +
                  ", you predicted that it is a \"" + classes[
                      int(CLASSIFICATION_RESULT["Prediction test"][0, index])].decode(
                     "utf-8") + "\" picture.")
            plt.show()
        else:
            print("There are only 50 pictures , try again -")
    except ValueError:
        print("try again - ")











import numpy as np
from random import choice
import matplotlib.pyplot as plt

def plot_perceptron(training_data, labels, w):
    
    # Plot data
    for point, label in zip(training_data, labels):
        plt.plot(point[0], point[1], "ro" if label > 0 else "bo")
    
    # Calculate decision boundary values
    xValues = np.array([0, 1]) # use formula to get y values
    yValues = -(w[0]/w[1]) * xValues + (-w[2]/w[1])
    
    # Plot decision boundary line
    plt.plot(xValues, yValues)   
        

def train_perceptron(training_data):
    '''
    Train a perceptron model given a set of training data
    :param training_data: A list of data points, where training_data[0]
    contains the data points and training_data[1] contains the labels.
    Labels are +1/-1.
    :return: learned model vector
    '''
    X = training_data[0]
    y = training_data[1]
    model_size = X.shape[1]
    w = np.zeros(model_size)#np.random.rand(model_size)
    iteration = 1
    
    max_iterations = 10000
    learning_rate = 0.1
    
    while True:
        misclassified = []
        for i in range( len(X) ):
            
            # compute results according to the hypothesis
            f = np.dot(w, X[i])
            yhat = 1 if f >= 0 else -1
            
            # get incorrect predictions (you can get the indices)
            if yhat != y[i]:
                misclassified.append((i, yhat))

        # Check the convergence criteria (if there are no misclassified
        # points, the PLA is converged and we can stop.)
        if len(misclassified) == 0:
            print("Iterations: ", iteration)
            return w

        # Pick one misclassified example.
        point = choice(misclassified)

        # Update the weight vector with perceptron update rule
        w += learning_rate * (y[ point[0]] - point[1]) * X[ point[0] ]

        iteration += 1
        if iteration >= max_iterations:
            break
    
    print("Iterations: ", iteration)
    return w

def print_prediction(model,data):
    '''
    Print the predictions given the dataset and the learned model.
    :param model: model vector
    :param data:  data points
    :return: nothing
    '''
    result = np.matmul(data,model)
    predictions = np.sign(result)
    for i in range(len(data)):
        print("{}: {} -> {}".format(data[i][:2], result[i], predictions[i]))


if __name__ == '__main__':
    
    rnd_x = np.array([[0,1,1],\
                      [0.6,0.6,1],\
                      [1,0,1],\
                      [1,1,1],\
                      [0.3,0.4,1],\
                      [0.2,0.3,1],\
                      [0.1,0.4,1],\
                      [0.5,-0.1,1]])

    rnd_y = np.array([1,1,1,1,-1,-1,-1,-1])
    rnd_data = [rnd_x,rnd_y]

    trained_model = train_perceptron(rnd_data)
    print("Model:", trained_model)
    plot_perceptron(rnd_x, rnd_y, trained_model)
    print_prediction(trained_model, rnd_x)




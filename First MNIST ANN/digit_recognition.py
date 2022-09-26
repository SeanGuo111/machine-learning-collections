#each layer output will be in rows
#each parameter matrix will contain its the weights for its previous layer in columns
#Avoiding need to transpose
# No need to do any unrolling nonsense in this, besides gradient check displaying
# Check out "view value in data viewer" upon right clicking variable watcher

# NOTE: This is similar to Andrew Ng's NN in most ways. It however uses ordinary gradient descent instead of fmincg optimizer, making it less accurate to a degree.

import numpy as np
import matplotlib.pyplot as plt

# SETUP FUNCTIONS ==============================================================================================================================================================================
def initialize_one_hot_encoded_y(x: np.ndarray):
    """Encodes the y values to binary vectors to represent categories."""
    array = np.zeros((x.shape[0], 10))
    for num in range(10):
        for indiv in range(500):
            array[(num*500) + indiv][num] = 1

    return array

def initialize_real_y():
    """Returns the real y values (non-binary). Just an array, not nd-array."""
    array = np.array([])
    for i in range(0, 10):
        array = np.concatenate((array, np.ones(500) * i))
    return array

def randomize_params(theta1, theta2) -> tuple[np.ndarray, np.ndarray]:
    """Randomizes the parameter matrices at the start."""
    epsilon = 0.12
    theta1_random = np.random.rand(theta1.shape[0], theta1.shape[1])
    theta1_random = theta1_random * (2 * epsilon) - epsilon # in range [-e, e]
    theta2_random = np.random.rand(theta2.shape[0], theta2.shape[1])
    theta2_random = theta2_random * (2 * epsilon) - epsilon # in range [-e, e]

    return theta1_random, theta2_random

def gradient_check(X, theta1, theta2, y, regularization_parameter):
    """Checks the gradient to make sure computation is correct."""

    # Approximated Gradient -----
    epsilon = 1 * (10 ** (-4))

    # Theta1
    approx_gradient_theta1 = np.zeros(theta1.shape)
    for row in range(10):
        # ^ Should be all (theta1.shape[0]), but only first 10 rows for sake of computational speed
        for column in range(theta1.shape[1]):
            theta1_above = np.copy(theta1)
            theta1_below = np.copy(theta1)
            theta1_above[row][column] += epsilon
            theta1_below[row][column] -= epsilon

            above_predictions = forward_propagation(X, theta1_above, theta2)[0]
            below_predictions = forward_propagation(X, theta1_below, theta2)[0]
            above_cost = cost_function(above_predictions, y, theta1_above, theta2, regularization_parameter)
            below_cost = cost_function(below_predictions, y, theta1_below, theta2, regularization_parameter)

            current_approx_gradient = (above_cost - below_cost) / (2 * epsilon)
            approx_gradient_theta1[row][column] = current_approx_gradient
            print(column + column * row)

    # Theta2
    approx_gradient_theta2 = np.zeros(theta2.shape)
    for row in range(theta2.shape[0]):
        for column in range(theta2.shape[1]):
            theta2_above = np.copy(theta2)
            theta2_below = np.copy(theta2)
            theta2_above[row][column] += epsilon
            theta2_below[row][column] -= epsilon

            above_predictions = forward_propagation(X, theta1, theta2_above)[0]
            below_predictions = forward_propagation(X, theta1, theta2_below)[0]
            above_cost = cost_function(above_predictions, y, theta1, theta2_above, regularization_parameter)
            below_cost = cost_function(below_predictions, y, theta1, theta2_below, regularization_parameter)

            current_approx_gradient = (above_cost - below_cost) / (2 * epsilon)
            approx_gradient_theta2[row][column] = current_approx_gradient
            print(column + column * row)
    
    # Backprop Gradient -----
    a3, a2, a1 = forward_propagation(X, theta1, theta2)
    backprop_gradient_theta1, backprop_gradient_theta2 = back_propagation(a1, a2, a3, y, theta1, theta2, regularization_parameter)
    
    # Comparison -----
    approx_gradient_theta1 = approx_gradient_theta1.reshape(-1,1)
    backprop_gradient_theta1 = backprop_gradient_theta1.reshape(-1,1)
    approx_gradient_theta2 = approx_gradient_theta2.reshape(-1,1)
    backprop_gradient_theta2 = backprop_gradient_theta2.reshape(-1,1)
    
    for i in range(100):
        print(f"{i+1}th Theta 1: Approx {approx_gradient_theta1[i][0]} | {backprop_gradient_theta1[i][0]} Backprop")
        print(f"{i+1}th Theta 2: Approx {approx_gradient_theta2[i][0]} | {backprop_gradient_theta2[i][0]} Backprop")


    theta1_difference = np.abs(approx_gradient_theta1[:100] - backprop_gradient_theta1[:100])
    theta1_difference_sum = np.sum(np.sum(theta1_difference))
    theta2_difference = np.abs(approx_gradient_theta2 - backprop_gradient_theta2)
    theta2_difference_sum = np.sum(np.sum(theta2_difference))

    print(f"First 100 Theta1 differences: {theta1_difference_sum}")
    print(f"All Theta2 differences: {theta2_difference_sum}")

def generate_data(X, y, real_y, percent_training, percent_validation, percent_testing, num_examples):
    """Generates data off of number of each example."""

    num_training = int(percent_training * num_examples)
    num_validation = int(percent_validation * num_examples)
    num_testing = int(percent_testing * num_examples)

    combined_data:np.ndarray = np.hstack((X, y, real_y))
    np.random.shuffle(combined_data)

    training_X = combined_data[:num_training,:400]
    training_y = combined_data[:num_training,400:410]
    training_real_y = combined_data[:num_training,410].tolist()

    validation_X = combined_data[num_training:num_training + num_validation,:400]
    validation_y = combined_data[num_training:num_training + num_validation,400:410]
    validation_real_y = combined_data[num_training:num_training + num_validation,410].tolist()

    testing_X = combined_data[num_training + num_validation:num_training + num_validation + num_testing,:400]
    testing_y = combined_data[num_training + num_validation:num_training + num_validation + num_testing,400:410]
    testing_real_y = combined_data[num_training + num_validation:num_training + num_validation + num_testing,410].tolist()

    return training_X, training_y, training_real_y, validation_X, validation_y, validation_real_y, testing_X, testing_y, testing_real_y

# ANALYSIS FUNCTIONS ==============================================================================================================================================================================
def sigmoid(x: np.ndarray):
    """Returns the sigmoid function of a matrix"""
    return 1 / (1 + np.exp(-1 * x))

def get_regularization_cost_term(m, theta1, theta2, regularization_parameter):
    """Returns the regularization term for the cost function, evaluated with the regularization parameter lambda"""
    theta1_sum = np.sum(theta1[1:] ** 2)
    theta2_sum = np.sum(theta2[1:] ** 2)
    regularization_term = (theta1_sum + theta2_sum) * regularization_parameter / (2 * m)
    return regularization_term


def cost_function(predictions:np.ndarray, y:np.ndarray, theta1: np.ndarray, theta2: np.ndarray, regularization_parameter):
    """Returns the cost for a certain set of parameter matrices"""
    m = y.shape[0]
    #predict() is predictions
    ifOne = -1 * y * np.log(predictions)
    ifZero = -1 * (1-y) * np.log(1 - predictions)
    total_offset = ifOne + ifZero

    #Sums every single element
    cost = np.sum(total_offset) / m

    #Regularize
    cost += get_regularization_cost_term(m, theta1, theta2, regularization_parameter)
    
    return cost

def forward_propagation(input: np.ndarray, theta1: np.ndarray, theta2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns 1st, the predictions for a certain set of parameter matrices, 2nd, the a2 value (hidden layer output), and 3rd, the a1 value (original X)"""
    
    #Hidden Layer
    ones_column = np.ones((input.shape[0], 1))
    a1 = np.hstack((ones_column, input))
    z2 = a1 @ theta1
    a2 = sigmoid(z2)

    #Output Layer
    #(Same ones column)
    a2 = np.hstack((ones_column, a2))
    z3 = a2 @ theta2
    a3 = sigmoid(z3)

    return a3, a2, a1


def back_propagation(a1: np.ndarray, a2: np.ndarray, a3: np.ndarray, y:np.ndarray, theta1: np.ndarray, theta2: np.ndarray, regularization_parameter) -> tuple[np.ndarray, np.ndarray]:
    """Calculates gradient of the cost with respect to both parameter matrices, for all training examples"""
    m = y.shape[0]
    theta1_training_ex_accumulator = np.zeros(theta1.shape)
    theta2_training_ex_accumulator = np.zeros(theta2.shape)
    
    for training_example in range(m):
        # DO THE BACKPROP
        
        # THETA 2 GRADIENT ------------------------------
        
        # Delta1 Calculation
        delta3 = np.array([a3[training_example] - y[training_example]]) # 1 x 10
        # Respect to Theta2 Connection Calculation
        pd_z3_respect_to_theta2 = a2[training_example].reshape(-1,1) # 26 x 1
        
        # Full pd cost/theta2
        theta2_current_gradient = pd_z3_respect_to_theta2 @ delta3 # 26 x 10
        theta2_training_ex_accumulator += theta2_current_gradient
        
        # THETA 1 GRADIENT ------------------------------

        # Delta2 Calculation (3 + 1 pieces: delta3, sigmoid, theta2, + a1)
        pd_z3_respect_to_a2 = ((theta2[1:].T)) # 10 x 25. Not 10 x 26, because bias 1 term is disregarded, it is artificially added and not produced by previous layer calculations. In other words, theta1 has no effect on the theta2 bias 1s.
        current_a2 = a2[training_example][1:] # 1 x 25, again bias 1 term disregarded due to being unaffected by theta2 
        sigmoid_deriv = (current_a2 * (1-current_a2)).reshape(1, -1) 

        delta2 = (delta3 @ pd_z3_respect_to_a2) * sigmoid_deriv
        # Respect to Theta1 Connection Calculation
        pd_z2_respect_to_theta1 = a1[training_example].reshape(-1,1) # 401 x 1
        
        # Full pd cost/theta1 
        theta1_current_gradient = pd_z2_respect_to_theta1 @ delta2
        theta1_training_ex_accumulator += theta1_current_gradient


    # Average, and regularize
    gradient_theta1 = theta1_training_ex_accumulator / m
    gradient_theta2 = theta2_training_ex_accumulator / m
    
    regularize_theta1 = np.concatenate((np.zeros((1, theta1.shape[1])), theta1[1:] * regularization_parameter / m), axis = 0)
    regularize_theta2 = np.concatenate((np.zeros((1, theta2.shape[1])), theta2[1:] * regularization_parameter / m), axis = 0)

    gradient_theta1 += regularize_theta1
    gradient_theta2 += regularize_theta2

    return gradient_theta1, gradient_theta2

def gradient_descent(training_X: np.ndarray, training_y: np.ndarray, validation_X: np.ndarray, validation_y: np.ndarray, theta1: np.ndarray, theta2: np.ndarray, learning_rate, epochs, regularization_parameter) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs through forwardprop and backprop, calculating cost, and optimizing training cost. Graphs both training and validation cost functions.
    
    This function returns 1. the training cost, 2. the validation cost, 3. theta1, and 4. theta2."""

    training_cost_history = []
    validation_cost_history = []

    # ONE GRADIENT DESCENT STEP
    for iteration in range(epochs):
        # Calculate training and validation costs
        a3, a2, a1 = forward_propagation(training_X, theta1, theta2)
        training_cost = cost_function(a3, training_y, theta1, theta2, regularization_parameter)
        validation_cost = cost_function(forward_propagation(validation_X, theta1, theta2)[0], validation_y, theta1, theta2, regularization_parameter)
       
        training_cost_history.append(training_cost)
        validation_cost_history.append(validation_cost)
        print(f"Iteration {iteration + 1} Cost | Training: {training_cost} | Validation: {validation_cost}")

        # Train
        gradient_theta_1, gradient_theta_2 = back_propagation(a1, a2, a3, training_y, theta1, theta2, regularization_parameter)

        theta1 -= learning_rate * gradient_theta_1
        theta2 -= learning_rate * gradient_theta_2

    final_training_cost = cost_function(forward_propagation(training_X, theta1, theta2)[0], training_y, theta1, theta2, regularization_parameter)
    final_validation_cost = cost_function(forward_propagation(validation_X, theta1, theta2)[0], validation_y, theta1, theta2, regularization_parameter)

    # Graph
    training_cost_history.append(final_training_cost)
    validation_cost_history.append(final_validation_cost)
    epoch_history = range(epochs + 1)
    graph_cost(epoch_history, training_cost_history, validation_cost_history, show_graph)

    return final_training_cost, final_validation_cost, theta1, theta2

# UTILITY FUNCTIONS ==============================================================================================================================================================================

def graph_cost(epochs, training_cost, validation_cost, show_graph):
    """Graphs the cost of the neural network."""
    if show_graph:
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.title("Training and Validation Cost vs. Epochs")

        plt.plot(epochs, training_cost, label = "Training Cost")
        plt.plot(epochs, validation_cost, label = "Validation Cost")
        plt.legend()
        plt.show()

def predict(input: np.ndarray, theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    """Returns the predicted integers for a certain input data matrix, whether it be one example or multiple, and parameters."""
    result = forward_propagation(input, theta1, theta2)[0]
    predictions: np.ndarray = np.argmax(result, axis=1) # axis 1 makes it return the greatest column index, for each row (axis 1 = columns)
    return predictions.astype(int)

def evaluate_accuracy(predictions: np.ndarray, y: np.ndarray, evaluate_missed):
    """Return percentage accuracy for some predictions and real values."""
    binary_equivalence_matrix:np.ndarray = np.equal(predictions, y) # true if equal, false if not
    total_correct = binary_equivalence_matrix.sum() # trues -> 1, falses -> 0
    percentage_correct = total_correct / predictions.size * 100

    if (evaluate_missed):
        incorrects = np.zeros(10)

        print("Each Miss ------------------------------")
        for i in range(binary_equivalence_matrix.size):
            if not binary_equivalence_matrix[i]:
                print(f"Missed #{i+1}: Guessed {predictions[i]} | Actually {y[i]}")
                incorrects[int(y[i])] += 1

        
        plt.bar(list(range(10)), incorrects, width = 0.4)
        plt.xlabel("Integer")
        plt.ylabel("Number of misses")
        plt.title("Most missed integers")
        plt.xticks(np.arange(0,10,1))
        plt.show()

    return percentage_correct

def play(X, theta1, theta2, y):
    random_index = np.random.randint(X.shape[0])
    random_example = np.reshape(X[random_index], (1,-1))

    pixels = random_example.reshape((20, 20)).T

    plt.imshow(pixels, cmap='gray')
    plt.pause(2)
    plt.close()

    random_correct = y[random_index]
    output = predict(random_example, theta1, theta2)[0]
    print(f"Integer: {random_correct} | Prediction: {output} | Correct: {random_correct == output}")


# HYPERPARAMETERS ==============================================================================================================================================================================

num_labels = 10
num_examples = 5000
percent_training = 0.7
percent_validation = 0.2
percent_testing = 0.1

num_hidden_nodes = 25
epochs = 750

learning_rate = 2.1
regularization_parameter = 0

show_graph = True # Global; can be accessed in gradient_descent()

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# EXECUTION OF CODE ==============================================================================================================================================================================
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# LOAD DATA
X: np.ndarray = np.loadtxt('C:\\Users\\swguo\\VSCode Projects\\Machine Learning\\Neural Networks\\First MNIST ANN\\digitsData.txt', delimiter = ',')
y = initialize_one_hot_encoded_y(X)
real_y = np.reshape(initialize_real_y(), (-1,1))

# INITIALIZE DATA
training_X, training_y, training_real_y, validation_X, validation_y, validation_real_y, testing_X, testing_y, testing_real_y = generate_data(X, y, real_y, percent_training, percent_validation, percent_testing, num_examples)

#INITIALIZE PARAMS
# theta1 = np.zeros((401, num_hidden_nodes))
# theta2 = np.zeros((num_hidden_nodes + 1, 10))
# theta1, theta2 = randomize_params(theta1, theta2)
theta1 = np.loadtxt("C:\\Users\\swguo\\VSCode Projects\\Machine Learning\\Neural Networks\\First MNIST ANN\\random_theta1.txt", delimiter=",")
theta2 = np.loadtxt("C:\\Users\\swguo\\VSCode Projects\\Machine Learning\\Neural Networks\\First MNIST ANN\\random_theta2.txt", delimiter=",")

#RUNNING ==============================================================================================================================================================================
# gradient_check(X, theta1, theta2, y, 1) Done

# Initial Stats ==========================================================================================================
initial_cost = cost_function(forward_propagation(testing_X, theta1, theta2)[0], testing_y, theta1, theta2, regularization_parameter)
initial_predictions = predict(testing_X, theta1, theta2)
initial_accuracy = evaluate_accuracy(initial_predictions, testing_real_y, False)
print(f"Initial Testing Cost: {initial_cost}")
print(f"Initial Testing Accuracy: {initial_accuracy}%")

# Optimize + Graph
#       TO TEST DATA:
#print("\nOPTIMIZE -----")
#final_training_cost, final_validation_cost, optimized_theta1, optimized_theta2 = gradient_descent(training_X, training_y, validation_X, validation_y, theta1, theta2, learning_rate, epochs, regularization_parameter)
#np.savetxt("C:\\Users\\swguo\\VSCode Projects\\Machine Learning\\Neural Networks\\First MNIST ANN\\optimal_theta1.txt", optimized_theta1, delimiter=",") # OVERWRITES OLD FILE
#np.savetxt("C:\\Users\\swguo\\VSCode Projects\\Machine Learning\\Neural Networks\\First MNIST ANN\\optimal_theta2.txt", optimized_theta2, delimiter=",")
#       TO LOAD DATA:
optimized_theta1 = np.loadtxt('C:\\Users\\swguo\\VSCode Projects\\Machine Learning\\Neural Networks\\First MNIST ANN\\optimal_theta1.txt', delimiter = ',')
optimized_theta2 = np.loadtxt('C:\\Users\\swguo\\VSCode Projects\\Machine Learning\\Neural Networks\\First MNIST ANN\\optimal_theta2.txt', delimiter = ',')

# Final Training Stats ==========================================================================================================
# Final Training Cost

final_training_cost = cost_function(forward_propagation(training_X, optimized_theta1, optimized_theta2)[0], training_y, optimized_theta1, optimized_theta2, regularization_parameter)
print(f"\nFinal Training Cost: {final_training_cost}")

# Evaluate Training Accuracy
final_training_predictions = predict(training_X, optimized_theta1, optimized_theta2)
final_training_accuracy = evaluate_accuracy(final_training_predictions, training_real_y, False)
print(f"Final Training Accuracy: {final_training_accuracy}%")


# Final Validation Stats ==========================================================================================================
# Final Validation Cost
final_validation_cost = cost_function(forward_propagation(validation_X, optimized_theta1, optimized_theta2)[0], validation_y, optimized_theta1, optimized_theta2, regularization_parameter)
print(f"\nFinal Validation Cost: {final_validation_cost}")

# Evaluate Validation Accuracy
final_validation_predictions = predict(validation_X, optimized_theta1, optimized_theta2)
final_validation_accuracy = evaluate_accuracy(final_validation_predictions, validation_real_y, False)
print(f"Final Validation Accuracy: {final_validation_accuracy}%")

# Final Testing Stats ==========================================================================================================
# Final Testing Cost
final_testing_cost = cost_function(forward_propagation(testing_X, optimized_theta1, optimized_theta2)[0], testing_y, optimized_theta1, optimized_theta2, regularization_parameter)
print(f"\nFinal Testing Cost: {final_testing_cost}")

final_testing_predictions = predict(testing_X, optimized_theta1, optimized_theta2)
final_testing_accuracy = evaluate_accuracy(final_testing_predictions, testing_real_y, False)
print(f"Final Testing Accuracy: {final_testing_accuracy}%")


# Test Playground ==========================================================================================================
play_bool = True
if play_bool:
    user_input = input("\nPlay? Press n to cancel: ")
    while (user_input != "n"):
        play(testing_X, optimized_theta1, optimized_theta2, testing_real_y)
        user_input = input("\nPlay? Press n to cancel: ")

print("\nFinished.")

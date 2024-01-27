import numpy as np
from data import get_mnist
import matplotlib.pyplot as plt



# Represents 1/1+e^-x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
"""
images, labels = get_mnist()


weights_from_input_layer_to_hidden_layer = np.random.uniform(-0.5, 0.5, (20, 784))
weights_from_hidden_layer_to_output_layer = np.random.uniform(-0.5, 0.5, (10, 20))
biases_from_input_layer_to_hidden_layer = np.zeros((20, 1))
biases_from_hidden_layer_to_output_layer = np.zeros((10, 1))

# how sporadically to change weights/biases, more = less overall accuracy, less = more overall accuracy
learn_rate = 0.01
correct_guesses = 0
epochs = 3
for epoch in range(epochs):
    for img, l in zip(images, labels):
        # turn vector into matrix to allow multiplications
        img.shape += (1,)
        l.shape += (1,)
        # Forward propagation input -> hidden
        h_pre = biases_from_input_layer_to_hidden_layer + weights_from_input_layer_to_hidden_layer @ img
        h = sigmoid(h_pre)
        # Forward propagation hidden -> output
        o_pre = biases_from_hidden_layer_to_output_layer + weights_from_hidden_layer_to_output_layer @ h
        o = sigmoid(o_pre)

        # Cost / Error calculation
        # o is the output, l is the label
        # So, o - l is the difference between the output and the label or actual value
        # 1/ 10 * np.sum((o - l) ** 2, axis=0) = 1/10 * (o1 - l1)^2 + (o2 - l2)^2 + ... + (o10 - l10)^2
        # axis=0 means that it will sum the columns, not the rows
        # e.g. [[1, 2, 3], [4, 5, 6]] -> [5, 7, 9]
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        # np.argmax returns the highest value (or answer) in the array
        correct_guesses += int(np.argmax(o) == np.argmax(l))

        # Backpropagation output -> hidden (cost function derivative)
        # needed only for output layer
        # delta_o can be calculated like this because of how the mean squared error function is defined
        delta_o = o - l
        weights_from_hidden_layer_to_output_layer += -learn_rate * delta_o @ np.transpose(h)
        biases_from_hidden_layer_to_output_layer += -learn_rate * delta_o

        # needed for all hidden layers
        # Backpropagation hidden -> input (activation function derivative)
        # (h * (1 - h)) is the derivative of the sigmoid function
        delta_h = np.transpose(weights_from_hidden_layer_to_output_layer) @ delta_o * (h * (1 - h))
        weights_from_input_layer_to_hidden_layer += -learn_rate * delta_h @ np.transpose(img)
        biases_from_input_layer_to_hidden_layer += -learn_rate * delta_h

    # Show accuracy for this epoch
    print(f"Acc: {round((correct_guesses / images.shape[0]) * 100, 2)}%")
    print(f"Err: {e}")
    correct_guesses = 0


def predict(img):
    img.shape += (1,)
    # Forward propagation input -> hidden
    # you don't apparently need to reshape the img
    hidden_output_pre = biases_from_input_layer_to_hidden_layer + weights_from_input_layer_to_hidden_layer @ img.reshape(
        784, 1)
    hidden_output = sigmoid(hidden_output_pre)
    # Forward propagation hidden -> output
    output_pre = biases_from_hidden_layer_to_output_layer + weights_from_hidden_layer_to_output_layer @ hidden_output
    output = sigmoid(output_pre)
    return output


# Show results
while True:
    index = int(input("Enter a number between 0 and 59999: "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    output = predict(img)
    plt.title(f"NN predicted its a {output.argmax()}")
    plt.show()

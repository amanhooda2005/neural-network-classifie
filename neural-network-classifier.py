import numpy as np
import matplotlib.pyplot as plt

letter_A = np.array([
    0, 1, 1, 1, 1, 0,
    1, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 1,
    1, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 1
]).reshape(5, 6)

letter_B = np.array([
    1, 1, 1, 1, 0, 0,
    1, 0, 0, 0, 1, 0,
    1, 1, 1, 1, 0, 0,
    1, 0, 0, 0, 1, 0,
    1, 1, 1, 1, 0, 0
]).reshape(5, 6)

letter_C = np.array([
    0, 1, 1, 1, 1, 0,
    1, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 1,
    0, 1, 1, 1, 1, 0
]).reshape(5, 6)

# Flatten for input to the neural network
training_data = np.array([
    letter_A.flatten(),
    letter_B.flatten(),
    letter_C.flatten()
])

target_output = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)    

        self.z2 = np.dot(self.a1, self.W2) + self.b2 
        self.a2 = self.sigmoid(self.z2)   
        return self.a2

    def backward(self, X, y, output):
        error_output = output - y
        d_output = error_output * self.sigmoid_derivative(self.z2)

        error_hidden = np.dot(d_output, self.W2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.z1)

        self.W2 -= self.learning_rate * np.dot(self.a1.T, d_output)
        self.b2 -= self.learning_rate * np.sum(d_output, axis=0, keepdims=True)
        self.W1 -= self.learning_rate * np.dot(X.T, d_hidden)
        self.b1 -= self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    def train(self, X, y, epochs):
        losses = []
        accuracies = []

        for epoch in range(epochs):
            output = self.forward(X)

            self.backward(X, y, output)

            loss = np.mean(np.square(y - output))
            losses.append(loss)

            predictions = np.argmax(output, axis=1)
            true_labels = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == true_labels)
            accuracies.append(accuracy)

            if epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return losses, accuracies

    def predict(self, X):
        return self.forward(X)

input_size = 30  
hidden_size = 10 
output_size = 3  
learning_rate = 0.5 
epochs = 5000 

nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

print("Starting training...")
losses, accuracies = nn.train(training_data, target_output, epochs)
print("Training complete.")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

def display_letter(letter_pixels, title="Input Letter"):
    plt.imshow(letter_pixels.reshape(5, 6), cmap='gray_r') 
    plt.title(title)
    plt.axis('off')
    plt.show()

test_A = letter_A.flatten().reshape(1, -1) 
prediction_A = nn.predict(test_A)
predicted_class_A_index = np.argmax(prediction_A)
class_labels = ['A', 'B', 'C']
predicted_class_A = class_labels[predicted_class_A_index]

print(f"\nPrediction for 'A': {prediction_A}")
print(f"The network predicts 'A' as: {predicted_class_A}")
display_letter(letter_A, f"Input Letter: A, Predicted: {predicted_class_A}")

test_B = letter_B.flatten().reshape(1, -1)
prediction_B = nn.predict(test_B)
predicted_class_B_index = np.argmax(prediction_B)
predicted_class_B = class_labels[predicted_class_B_index]

print(f"\nPrediction for 'B': {prediction_B}")
print(f"The network predicts 'B' as: {predicted_class_B}")
display_letter(letter_B, f"Input Letter: B, Predicted: {predicted_class_B}")

test_C = letter_C.flatten().reshape(1, -1)
prediction_C = nn.predict(test_C)
predicted_class_C_index = np.argmax(prediction_C)
predicted_class_C = class_labels[predicted_class_C_index]

print(f"\nPrediction for 'C': {prediction_C}")
print(f"The network predicts 'C' as: {predicted_class_C}")
display_letter(letter_C, f"Input Letter: C, Predicted: {predicted_class_C}")
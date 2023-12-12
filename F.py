import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tkinter import Tk, Label, Button, StringVar, Entry
import numpy as np

# Convert the TensorFlow MNIST data to PyTorch DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

# Define the PyTorch neural network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Instantiate the PyTorch neural network and define the loss and optimizer
model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the PyTorch neural network
def train_pytorch(epochs):
    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in train_loader:
            data = data.view(-1, 784)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print('Epoch', epoch, 'loss:', epoch_loss)

train_pytorch(15)

# Testing the PyTorch neural network
def test_pytorch():
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 784)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    print('Accuracy:', accuracy)

test_pytorch()

# GUI setup for testing the PyTorch neural network
class NeuralNetworkGUI:
    def __init__(self, master):
        self.master = master
        master.title("PyTorch Neural Network GUI")

        self.label = Label(master, text="Enter a number for testing:")
        self.label.pack()

        self.entry = Entry(master)
        self.entry.pack()

        self.result_var = StringVar()
        self.result_label = Label(master, textvariable=self.result_var)
        self.result_label.pack()

        self.test_button = Button(master, text="Test", command=self.test_network)
        self.test_button.pack()

    def test_network(self):
        input_str = self.entry.get()
        input_list = np.array(list(map(float, input_str)))
        input_tensor = torch.Tensor(input_list).view(1, -1)
        output = model(input_tensor)
        predicted_number = torch.argmax(output).item()
        self.result_var.set(f"Predicted Number: {predicted_number}")

# Create GUI window
root = Tk()
neural_network_gui = NeuralNetworkGUI(root)
root.mainloop()

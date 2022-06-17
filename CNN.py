import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class CNN(nn.Module):
    # constructor to create CNN instance
    def __init__(self, num_classes=10):
        # call the parent class explicitly to initialize the nn.Module
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=12, stride=2, padding=0),
            nn.BatchNorm2d(25),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dense = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),  # each unit in max pool connected to 1024 units of fully connected layer
            nn.ReLU(),
            nn.Linear(1024, num_classes)  # connect to 10 unit fully connected layer
        )
        self.type = 'CNN'

    def forward(self, x):
        x = self.conv1(x)  # shape [50, 25, 9, 9]
        x = self.conv2(x)  # shape [50, 64, 4, 4]
        # flatten the max_pool layer to 1 dimension
        x = x.view(-1, 64 * 4 * 4)  # shape [50, 1024]
        out = self.dense(x)

        return out


# data loader to prepare MNIST dataset
def create_dataloader():
    # download the MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='data',
                                               train=True,
                                               download=True,
                                               transform=transforms.ToTensor())  # transform the data into 28*28*1

    test_dataset = torchvision.datasets.MNIST(root='data',
                                              train=False,
                                              download=True,
                                              transform=transforms.ToTensor())  # transform the data into 28*28*1
    # Data loader
    # pass training dataset as batch and reshuffle the data at every epoch
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=50,
                                               shuffle=True)
    # pass test dataset as batch
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=50,
                                              shuffle=False)

    return train_loader, test_loader


# train the model to estimate the parameters
def train(train_loader, model, criterion, optimizer, num_epochs):
    # total step = no. of images / minibatch size ==> 60000/50 = 1200
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            if model.type == "MLP":
                images = images.reshape(-1, 28 * 28)

            # forward pass of the model
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward and optimize
            # clear all the gradients to zero in the optimizer
            optimizer.zero_grad()
            # propagate the loss in backward direction
            loss.backward()
            # update the model parameters
            optimizer.step()

            # keep track the accuracy every 100 iterations
            if (step + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, step + 1, total_step, loss.item()))


# validate the model
def test(test_loader, model):
    # disable the gradient computation since the model is trained
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            if model.type == 'MLP':
                images = images.reshape(-1, 28 * 28)

            # outputs are tensor [50, 10] representing the likelihood of each class
            outputs = model(images)

            # return the index of the class with the highest likelihood
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the network: {} %'.format(100 * correct / total))

# get all convolution layers and the corresponding weights
def getConvLayer(model):
    # counter = 0  # count the number of conv layer
    model_weights = []  # store the weights of each conv layer
    conv_layers = []  # store the conv layers
    model_children = list(model.children())  # get the children of the model

    # iterate over the children of the model
    for child in model_children:
        # check if any direct child is conv layer
        if type(child) == nn.Conv2d:
            # counter += 1
            model_weights.append(child.weight)
            conv_layers.append(child)

        # check if any conv layer in the nn.sequential block
        elif type(child) == nn.Sequential:
            for layer in child.children():
                if type(layer) == nn.Conv2d:
                    # counter += 1
                    model_weights.append(layer.weight)
                    conv_layers.append(layer)
    # print(f"Total convolutional layers: {counter}")
    return model_weights, conv_layers

# visualize the filters of conv layer
def visualizeConvFilter(model_weights):
    # plot figure of filters
    plt.figure(figsize=(18, 16))

    for i, filter in enumerate(model_weights[0]):
        # plot 25 filters in 5 rows, 5 cols
        plt.subplot(5, 5, i + 1)  # plot number starts at 1

        # detach the tensor from the graph, move to cpu and then convert to numpy
        plt.imshow(filter[0, :, :].detach().cpu().numpy(), cmap='gray')

        plt.axis('off')

        # save the figure
        plt.savefig('filter1.png')

    plt.show()


if __name__ == "__main__":
    # configure the device of a tensor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prepare dataset and create dataloader
    train_loader, test_loader = create_dataloader()

    # instantiate convolution neural network instance
    model = CNN()

    # move the model to the selected device
    model.to(device)

    # apply the cross entropy loss function
    criterion = nn.CrossEntropyLoss()

    # set the learning rate to be 1e-4 in optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # train the model by epochs = 4 to have iterations between 1000 and 5000
    train(train_loader, model, criterion, optimizer, num_epochs=4)

    # validate the model
    test(test_loader, model)

    # retrieve all convolution layers and weights
    model_weights, conv_layers = getConvLayer(model)

    # visualize the filters in the first conv layer
    visualizeConvFilter(model_weights)

    """
    # show the conv layer info
    for i, (weight, conv) in enumerate(zip(model_weights, conv_layers)):
        print(f"Conv layer {i+1}: {conv}, shape: {weight.shape}")
    """






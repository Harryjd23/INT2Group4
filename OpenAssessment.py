import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch import optim
from torchvision import transforms

import sys

if "--unsafe" in sys.argv:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context


DATA_LOCATION = "files"


num_epochs = 20
batch_size_train = 64
batch_size_test = 64



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(3, 256, kernel_size = 5,
                               stride = (1,1), padding =(1,1))
        self.conv2 = nn.Conv2d(256, 256, kernel_size = 5,
                               stride = (1,1), padding =(1,1))
        self.conv3 = nn.Conv2d(256, 256, kernel_size = 5,
                               stride = (1,1), padding =(1,1))
        self.layer1 = nn.BatchNorm2d(256)
        self.layer2 = nn.BatchNorm2d(256)
        self.layer3 = nn.BatchNorm2d(256)
        self.pool1 = nn.MaxPool2d(3,3)
        self.fc1 = nn.Linear(4096,1000)
        self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(500,102)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.layer1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.layer2(x))
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.Flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# define your loss function
criterion = torch.nn.CrossEntropyLoss()

# define your optimizer
network = Net()
optimizer = torch.optim.Adam(network.parameters(), lr=0.00001)


transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64))])  # Resize the images to a consistent size

init_train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.Flowers102(DATA_LOCATION,split = "train", download=True,
                             transform=transformer
                             ),
  batch_size=batch_size_train, shuffle=True)


validation_loader = torch.utils.data.DataLoader(
    torchvision.datasets.Flowers102(DATA_LOCATION, split = "val", download=True,
                                transform=transformer
                                ),
    batch_size=batch_size_test, shuffle=False)


# The test set is to be used only for the final evaluation
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.Flowers102(DATA_LOCATION, split = "test", download=True,
                             transform=transformer
                             ),
 batch_size=batch_size_test, shuffle=False)


mean = 0 
s_deviation = 0
for images, _ in init_train_loader:
    batch = images.size(0)
    images = images.view(batch, images.size(1), -1)
    mean+= images.mean(2).sum(0)
    s_deviation+= images.std(2).sum(0)

mean /= len(init_train_loader.dataset)
s_deviation /= len(init_train_loader.dataset)

transformer_augmenter = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.2),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomRotation(55),
    transforms.Resize((64, 64)),  # Resize the images to a consistent size,
    transforms.Normalize(mean = mean, std = s_deviation)])  # Normalize the image tensors

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.Flowers102(DATA_LOCATION,split = "train", download=True,
                             transform=transformer_augmenter
                             ),
  batch_size=batch_size_train, shuffle=True)



def validate():
    """
    Used to check how the model is performing while it is training.
    """
    # set the model to evaluation mode
    network.eval()

    # keep track of the total number of correct predictions
    total_correct = 0
    total_samples = 0

    # loop over the batches in the validation data
    for data, target in validation_loader:

        # forward pass through the model
        output = network(data)

        # get the predicted class for each example in the batch
        _, predicted = torch.max(output, 1)

        # calculate the number of correct predictions in this batch
        batch_correct = (predicted == target).sum().item()
        # add the number of correct predictions in this batch to the total
        total_correct += batch_correct
        total_samples += target.size(0)

    # calculate the overall accuracy
    accuracy = 100 * (total_correct / total_samples)

    print('Validation accuracy: {:.4f}%'.format(accuracy))
    return accuracy


def test():
    """
    **ONLY USE AFTER FINISHING TRAINGING.**
    This is used to evaludate the model after it has finished training.
    """

    network.eval()

    # keep track of the total number of correct predictions
    total_correct = 0
    total_samples = 0

    for data, target in test_loader:

        # forward pass through the model
        output = network(data)

        # get the predicted class for each example in the batch
        _, predicted = torch.max(output, 1)

        # calculate the number of correct predictions in this batch
        batch_correct = (predicted == target).sum().item()
        # add the number of correct predictions in this batch to the total
        total_correct += batch_correct
        total_samples += target.size(0)

    # calculate the overall accuracy
    accuracy = 100 * (total_correct / total_samples)

    print('Test accuracy: {:.4f}%'.format(accuracy))
    return accuracy


def train():
    # loop over the epochs
    for epoch in range(num_epochs):

        # set the model to train mode
        network.train()

        # loop over the batches in the training data
        for batch_idx, (data, target) in enumerate(train_loader):

            # zero the gradients for this batch
            optimizer.zero_grad()

            # forward pass through the model
            output = network(data)

            # calculate the loss
            loss = criterion(output, target)

            # backward pass through the model to calculate the gradients
            loss.backward()

            # update the parameters using the gradients and optimizer
            optimizer.step()

            # print the loss for every 10 batches
            if batch_idx % 5 == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, batch_idx, len(train_loader), loss.item()))
                

        if epoch % 2 == 0:
            validate()


    print(f"Training accuracy:{round(validate(), 4)}%")



def main():
    print("\n== Starting training. ==")
    train()
    print("== Finished training. ==")
    accuracy = test()
    print(f"Final Accuracy:{round(accuracy, 4)}%")
    print("== Finished testing. ==")


if __name__ == "__main__":
    main()
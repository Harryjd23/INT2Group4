import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import sys

# Prevents Errors?
if "--unsafe" in sys.argv:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

# Force to use CPU instead of RAM?
FORCE_CPU = False
if "--cpu" in sys.argv:
    FORCE_CPU = True

# Location to load datasets
DATA_LOCATION = "files"

# Image Dimensions for Transformations
IMAGE_W = 128
IMAGE_H = 128
IMAGE_MAT_WIDTH = 65536

# Hyperparameters
n_epochs = 125
batch_size_train = 32
batch_size_test = 32

# Calculates square dimensions for best image
def dims_to_square(x: int, y: int) -> tuple[int, int, int, int]:
    if x > y:
        return ((x-y)/2, 0, y, y)
    else:
        return (0, (y-x)/2, x, x)


# Crop sides (or top/bottom) of an image to make it square.
class Squarer:
    def __init__(self):
        pass

    def __call__(self, tensor):
        x, y = tensor.size[-2:]  # Get the last two dimensions of the tensor.
        return transforms.functional.crop(tensor, *dims_to_square(x, y))

# Test if GPU is available, if not CPU is used for computations
device = "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"


# Neural Network Definition 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)  # Convolution 1
        self.bn1 = nn.BatchNorm2d(128)  # Batch Normalization 1
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(1024)
        self.prelu = nn.PReLU() # Instantiates PReLU activation function
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces Spatial Dimensions
        self.fc1 = nn.Linear(IMAGE_MAT_WIDTH, 4096)  # Fully Connected layer 1
        self.fc2 = nn.Linear(4096,102)

    def forward(self, x):
        x = self.conv1(x) # Applies Convolutional Layer 1 to Tensor
        x = self.prelu(self.bn1(x)) # Applies PReLU function to Batch Normal Layer, then Tensor
        x = self.pool(x) # Applies pool function to Tensor
        x = self.conv2(x)
        x = self.prelu(self.bn2(x))
        x = self.pool(x)
        x = self.conv3(x)
        x = self.prelu(self.bn3(x))
        x = self.pool(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flattens Tensor and preserves batch size
        x = self.fc1(x) # Applies Fully Connected Layer 1 to Tensor
        x = self.fc2(x)

        return x # Returns Tensor

# Define loss function, Neural Network, Optimizer & Scheduler
criterion = torch.nn.CrossEntropyLoss()
network = Net().to(device)
optimizer = optim.Adam(network.parameters(), lr=0.00001, weight_decay = 0.01)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=20)

# Define Transformations for Test/Validation Dataset
transform1 = transforms.Compose([
    Squarer(), # Sets images to a square with the max amount of the image within possible
    transforms.Resize((IMAGE_W, IMAGE_H)), # Resize the images to a consistent size
    transforms.ToTensor()]) # Converts PIL to Pytorch Tensor

# Define Transformations for Training Dataset
transform2 = transforms.Compose([
    transforms.RandomRotation(25), # Rotates random images 25%
    transforms.RandomHorizontalFlip(0.3), # Flips random 30% of images Horizontally 
    Squarer(),
    transforms.Resize((IMAGE_W, IMAGE_H)),
    transforms.ToTensor()])

# Load Training Dataset
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.Flowers102(DATA_LOCATION,split = "train", download=True,
                             transform= transform2
                             ),
  batch_size=batch_size_train, shuffle=True)

# Load Validation Dataset
val_loader = torch.utils.data.DataLoader(
  torchvision.datasets.Flowers102(DATA_LOCATION, split = "val", download=True,
                             transform= transform1
                             ),

 batch_size=batch_size_test, shuffle=False)

# Load the Testing Dataset
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.Flowers102(DATA_LOCATION, split = "test", download=True,
                             transform= transform1
                             ),

 batch_size=batch_size_test, shuffle=False)


# Loop over the epochs
def train(epoch):
    train_loss = 0
    correct = 0
    total = 0
    # Set the model to train mode
    network.train()

    # Loop over the batches in the training data
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        # forward pass through the model
        output = network(data)

        # calculate the loss
        loss = criterion(output, target)

        # Zero the gradients
        optimizer.zero_grad()

        # backward pass through the model to calculate the gradients
        loss.backward()
        optimizer.step()


        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1) # Gets the image predictions

        total += target.size(0) # Gets the total images in Training Dataset

        correct += (predicted == target).sum().item() # Sums all correct predictions
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, i*len(data), len(train_loader.dataset),100* i / len(train_loader),
        loss))
    accuracy = 100 * correct / total
    loss = train_loss / len(train_loader)
    scheduler.step()
    return loss, accuracy

def validate():
    network.eval() # Set the model to evaluate Mode
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            loss = criterion(output, target) 
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
        accuracy = 100 * correct / total
        loss = val_loss / total
        return loss, accuracy


def test():
    # set the model to evaluation mode
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
        accuracy = 100 * correct / total
        loss = test_loss / total

        print('Test accuracy: {:.4f}%'.format(accuracy))
        return loss, accuracy


def main():
    best_acc = 0

    for epoch in range(n_epochs+1):
        train_loss, train_acc = train(epoch)
        if epoch % 1 == 0:
            valid_loss, valid_acc = validate()
            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(network.state_dict(), 'Model.pth') # Saves the model
                network.load_state_dict(torch.load("Model.pth")) # Loads model with saved weights and parameters
        print(f'Epoch [{epoch}/{n_epochs}], Train Loss: {train_loss:.4f}, Valid Acc: {valid_acc:.2f}%')
    test()


if __name__ == '__main__':
    # This means we can import the neural network without running training, so we can generate pictures of it.
    main()
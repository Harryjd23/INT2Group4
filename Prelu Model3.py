import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import sys

if "--unsafe" in sys.argv:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

FORCE_CPU = False
if "--cpu" in sys.argv:
    FORCE_CPU = True


DATA_LOCATION = "files"

IMAGE_W = 128
IMAGE_H = 128
IMAGE_MAT_WIDTH = 65536

n_epochs = 125
batch_size_train = 32
batch_size_test = 32


def dims_to_square(x: int, y: int) -> tuple[int, int, int, int]:
    if x > y:
        return ((x-y)/2, 0, y, y)
    else:
        return (0, (y-x)/2, x, x)


"""Crop sides (or top/bottom) of an image to make it square."""
class Squarer:
    def __init__(self):
        pass

    def __call__(self, tensor):
        x, y = tensor.size[-2:]  # Get the last two dimensions of the tensor.
        return transforms.functional.crop(tensor, *dims_to_square(x, y))


device = "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)  # Convolution 1.
        self.bn1 = nn.BatchNorm2d(128)  # Batch Normalization 1.
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(1024)
        self.prelu = nn.PReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(IMAGE_MAT_WIDTH, 4096)  # Fully Connected layer 1.
        self.fc2 = nn.Linear(4096,102)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu(self.bn1(x))
        x = self.pool(x)
        x = self.conv2(x)
        x = self.prelu(self.bn2(x))
        x = self.pool(x)
        x = self.conv3(x)
        x = self.prelu(self.bn3(x))
        x = self.pool(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

# define your loss function
criterion = torch.nn.CrossEntropyLoss()

# define your optimizer
network = Net().to(device)
optimizer = optim.Adam(network.parameters(), lr=0.00001, weight_decay = 0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.999)

transform1 = transforms.Compose([
    Squarer(),
    transforms.Resize((IMAGE_W, IMAGE_H)),
    transforms.ToTensor()])  # Resize the images to a consistent size

transform2 = transforms.Compose([
    transforms.RandomRotation(25),
    transforms.RandomHorizontalFlip(0.3),
    Squarer(),
    transforms.Resize((IMAGE_W, IMAGE_H)),
    transforms.ToTensor()])


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.Flowers102(DATA_LOCATION,split = "train", download=True,
                             transform= transform2
                             ),
  batch_size=batch_size_train, shuffle=True)

val_loader = torch.utils.data.DataLoader(
  torchvision.datasets.Flowers102(DATA_LOCATION, split = "val", download=True,
                             transform= transform1
                             ),

 batch_size=batch_size_test, shuffle=False)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.Flowers102(DATA_LOCATION, split = "test", download=True,
                             transform= transform1
                             ),

 batch_size=batch_size_test, shuffle=False)


# loop over the epochs
def train(epoch):
    train_loss = 0
    correct = 0
    total = 0
    # set the model to train mode
    network.train()

    # loop over the batches in the training data
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        # zero the gradients for this batch

        # forward pass through the model
        output = network(data)

        # calculate the loss
        loss = criterion(output, target)

        # backward pass through the model to calculate the gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, i*len(data), len(train_loader.dataset),100* i / len(train_loader),
        loss))
    print(correct,total)
    accuracy = 100 * correct / total
    loss = train_loss / len(train_loader)
    scheduler.step()
    return loss, accuracy

def validate():
    network.eval()
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
                torch.save(network.state_dict(), 'BestModel.pth')
                network.load_state_dict(torch.load("BestModel.pth"))
        print(f'Epoch [{epoch}/{n_epochs}], Train Loss: {train_loss:.4f}, Valid Acc: {valid_acc:.2f}%')
    test()


if __name__ == '__main__':
    # This means we can import the neural network without running training, so we can generate pictures of it.
    main()

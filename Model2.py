import torch
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


DATA_LOCATION = "files"

n_epochs = 50
batch_size_train = 32
batch_size_test = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32768, 4096)
        self.fc2 = nn.Linear(4096, 102)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# define your loss function
criterion = torch.nn.CrossEntropyLoss()

# define your optimizer
network = Net().to(device)
optimizer = optim.Adam(network.parameters(), lr=0.00001, weight_decay = 0.01)
best_acc = 0

transform1= transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64))])
"""transforms.Normalize(mean =[0.4330, 0.3819, 0.2964], std= [0.2613, 0.2123, 0.2239])])"""  # Resize the images to a consistent size

transform2 = transforms.Compose([
    transforms.RandomRotation(25),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((64,64)),
    transforms.ToTensor()])
"""transforms.Normalize(mean =[0.4330, 0.3819, 0.2964], std= [0.2613, 0.2123, 0.2239])])"""

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

for epoch in range(n_epochs):
    train_loss, train_acc = train(epoch)
    if epoch % 1 == 0:
        valid_loss, valid_acc = validate()
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(network.state_dict(), 'BestModel.pth')
            network.load_state_dict(torch.load("BestModel.pth"))
    print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_loss:.4f}, Valid Acc: {valid_acc:.2f}%')
test()

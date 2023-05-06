import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
n_epochs = 15
batch_size_train = 64
batch_size_test = 64
learning_rate = 0.00005
log_interval = 10


transform1= transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64))])  # Resize the images to a consistent size

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.Flowers102('/files/',split = "train", download=True,
                             transform= transform1
                             ),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.Flowers102('/files/', split = "test", download=True,
                             transform= transform1
                             ),

 batch_size=batch_size_test, shuffle=False)


mean = 0 
s_deviation = 0
for images, _ in train_loader:
    batch = images.size(0)
    images = images.view(batch, images.size(1), -1)
    mean+= images.mean(2).sum(0)
    s_deviation+= images.std(2).sum(0)

mean /= len(train_loader.dataset)
s_deviation /= len(train_loader.dataset)

transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.2),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomRotation(55),
    transforms.Resize((64, 64)),  # Resize the images to a consistent size,
    transforms.Normalize(mean = mean, std = s_deviation)  # Normalize the image tensors
])

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.Flowers102('/files/',split = "train", download=True,
                             transform= transform2
                             ),
  batch_size=batch_size_train, shuffle=True)




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
        x = F.relu(self.layer3(x))
        x = self.Flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

network = Net().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=learning_rate)
"""scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)"""
best_acc = 0
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def test():
  network.eval()
  with torch.no_grad():
    correct = 0
    samples = 0
    n_correct = [0 for i in range(102)]
    n_samples = [0 for i in range(102)]
    for images, labels in test_loader:
       images = images.to(device)
       labels = labels.to(device)
       output = network(images)
       _, predicted = torch.max(output,1)
       samples += labels.size(0)
       correct += (predicted == labels).sum().item() 
       for i in range(len(labels)):
            pred = predicted[i]
            label = labels[i]
            if label == pred:
                n_correct[label] +=1
            n_samples[label] +=1
    accuracy = 100.0 * correct / samples
    print('Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_loader.dataset), accuracy))
    network.train()
    return accuracy
    
for epoch in range(n_epochs):
    for i, (images, image_labels) in enumerate(train_loader):
            images = images.to(device)
            image_labels = image_labels.to(device)
            label_pred = network(images)
            loss = loss_func(label_pred, image_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            """print("Epoch Number =" + str(epoch)+ "Index = ", str(i), "/", str(len(train_loader)-1), "loss" + str(loss.item()))"""
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(images), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()))
            if i % log_interval == 0:
                train_losses.append(loss.item())
                train_counter.append(
                (i*64) + ((epoch-1)*len(train_loader.dataset)))

    current_acc = test() 
    if current_acc > best_acc:
        best_acc = current_acc
        torch.save(network.state_dict(), "model.pth")       

             
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

TRAIN_DIR = 'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
VALID_DIR = 'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid'

BATCH_SIZE = 4
LEARNING_RATE = 0.01
EPOCHS = 1 
MODEL_SAVE_PATH = 'model8.pth'

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(128 * 32 * 32, 512) 
        self.fc2 = nn.Linear(512, 38) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Загрузка данных...")
    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = torchvision.datasets.ImageFolder(VALID_DIR, transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    classes = train_dataset.classes
    print(f"Найдено классов: {len(classes)}")

    net = Net()
    net = nn.DataParallel(net).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

    print("Начало обучения...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    
                print(f'[Эпоха {epoch + 1}, Батч {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Обучение завершено!')

    torch.save(net.state_dict(), MODEL_SAVE_PATH)
    print(f'Веса модели сохранены в {MODEL_SAVE_PATH}')

    print("Оценка точности модели...")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Точность модели на тестовых изображениях: {100 * correct // total} %')

if __name__ == '__main__':
    main()
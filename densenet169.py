import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

data_dir = './flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

data_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet169(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(1664, 832),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(832, 416),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(416, 208),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(208, 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.01)
model.to(device)

epochs = 30
min_valid_loss = np.Inf
for e in range(epochs):
    print(e)
    trainloss = 0
    model.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        trainloss += loss.item()

    validloss = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            validloss += loss.item()

    if validloss < min_valid_loss:
        torch.save(model.state_dict(), 'checkpoint.pth')
        min_valid_loss = validloss

    print(f'trainloss: {trainloss/len(trainloader)}, testloss: {testloss/len(testloader)}')


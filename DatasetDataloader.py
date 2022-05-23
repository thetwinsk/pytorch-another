import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

print(len(train_set))

print(train_set.targets)

print(train_set.targets.bincount())

sample = next(iter(train_set))
print(len(sample))

image, label = sample

print(type(image))

print(type(label))

print(image.shape)

print(torch.tensor(label).shape)

print(image.squeeze().shape)

plt.imshow(image.squeeze(), cmap="gray")
print(torch.tensor(label))

display_loader = torch.utils.data.DataLoader(
    train_set, batch_size=10
)
batch = next(iter(display_loader))
print('len:', len(batch))

images, labels = batch
print('types:', type(images), type(labels))
print('shapes:', images.shape, labels.shape)

grid = torchvision.utils.make_grid(images, nrow=10)
plt.figure(figsize=(15, 15))
plt.imshow(np.transpose(grid, (1, 2, 0)))

print('labels:', labels)

grid = torchvision.utils.make_grid(images, nrow=10)

plt.figure(figsize=(15,15))
plt.imshow(grid.permute(1,2,0))

print('labels:', labels)

how_many_to_plot = 20

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=1, shuffle=True
)

plt.figure(figsize=(50,50))
for i, batch in enumerate(train_loader, start=1):
    image, label = batch
    plt.subplot(10,10,i)
    plt.imshow(image.reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.title(train_set.classes[label.item()], fontsize=28)
    if (i >= how_many_to_plot): break
plt.show()




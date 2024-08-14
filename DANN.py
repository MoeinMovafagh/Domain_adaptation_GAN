import torch
import torch.nn as nn 
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])




import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class BalancedDataset(Dataset):
    def __init__(self, image_dir, max_num_samples, transform=None):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

        # Assign labels before oversampling
        half_point = len(self.images) // 2
        self.labels = [0 if i < half_point else 1 for i in range(len(self.images))]

        # Oversampling the dataset
        self.num_samples = max_num_samples
        self.indices = np.tile(np.arange(len(self.images)), (self.num_samples // len(self.images) + 1))[:self.num_samples]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[self.indices[idx]])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        # Get the label for the oversampled dataset
        label = self.labels[self.indices[idx] % len(self.labels)]

        return image, label
    


class BalancedDataset_real(Dataset):
    def __init__(self, image_dir, max_num_samples, transform=None):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

        # Assign labels before oversampling
        half_point = len(self.images) // 2
        self.labels = [0, 1, 0, 0, 1, 0, 0, 0, 1,1,0]

        # Oversampling the dataset
        self.num_samples = max_num_samples
        self.indices = np.tile(np.arange(len(self.images)), (self.num_samples // len(self.images) + 1))[:self.num_samples]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[self.indices[idx]])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        # Get the label for the oversampled dataset
        label = self.labels[self.indices[idx] % len(self.labels)]

        return image, label    

# Example usage
directory_real = 'C:\\Users\\movafagm\\moein\\Include\\Real_data_angle_DANN'
directory_sim = 'C:\\Users\\movafagm\\moein\\Include\\Input_angle'

max_num_samples = 7000  # Size of the larger dataset

# Instantiate the datasets
real_dataset = BalancedDataset_real(directory_real, max_num_samples, transform=transform)
sim_dataset = BalancedDataset(directory_sim, max_num_samples, transform=transform)
print(len(real_dataset))
print(len(sim_dataset))
# Create data loaders
from torch.utils.data import DataLoader
real_loader = DataLoader(real_dataset, batch_size=32, shuffle=True)
sim_loader = DataLoader(sim_dataset, batch_size=32, shuffle=True)


import torch
import torch.nn as nn

import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Load a pre-trained ResNet and remove its final layer
        base_model = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return x

lay = 512



import torch
import torch.nn as nn

class FeatureExtractor1(nn.Module):
    def __init__(self):
        super(FeatureExtractor1, self).__init__()
        # Adjust the in_channels to 3 for RGB images
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False)
        )
        # The flattened size calculation depends on the final output size of the conv layer
        self.flattened_size = 256 * (256 // 16) * (256 // 16)  # Update this if the size calculation changes
        # Define the final fully connected layer
        self.fc = nn.Linear(self.flattened_size, 2048)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.flattened_size)  # Flatten the features
        x = self.fc(x)
        return x


# Assuming a grayscale image, hence in_channels=1. Change it accordingly for RGB images.




# Define the classifier networks
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(lay, 256)  # Adjusted input size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)  # Additional layer
        self.fc5 = nn.Linear(32, 16)  # Additional layer
        self.fc6 = nn.Linear(16, 1)   # Final layer
        self.relu = nn.LeakyReLU(inplace=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.sig(x)
        return x




# Define the classifier networks
class Classifier_shallow(nn.Module):
    def __init__(self):
        super(Classifier_shallow, self).__init__()
        self.fc1 = nn.Linear(lay, 256)  # Adjusted input size
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.95)  # Dropout layer with 50% probability

        self.relu = nn.LeakyReLU(inplace=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sig(x)
        return x



# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(lay, 256)  # Adjusted input size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)  # Additional layer
        self.fc5 = nn.Linear(32, 16)  # Additional layer
        self.fc6 = nn.Linear(16, 1)   # Final layer
        self.relu = nn.LeakyReLU(inplace=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.sig(x)
        return x

import numpy as np

class CustomBCELoss(nn.Module):
    def __init__(self):
        super(CustomBCELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Ensuring that predictions are within (0,1) to avoid log(0) error
        y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)

        # Binary cross-entropy loss calculation
        loss = -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        return loss




device = torch.device("cuda")
# Instantiate the networks
def discriminator_accuracy(discriminator, real_data, fake_data):
    # Real data prediction
    real_pred = discriminator(real_data)
    real_correct = (real_pred > 0.5).sum().item()  # Count how many are correctly identified as real

    # Fake data prediction
    fake_pred = discriminator(fake_data)
    fake_correct = (fake_pred < 0.5).sum().item()  # Count how many are correctly identified as fake

    # Calculate accuracy
    total = real_data.size(0) + fake_data.size(0)  # Total number of data points
    total_correct = real_correct + fake_correct
    accuracy = total_correct / total

    return accuracy

feature_extractor_real = FeatureExtractor().to(device)
feature_extractor_sim = FeatureExtractor().to(device)
classifier_real = Classifier().to(device)
classifier_sim = Classifier().to(device)
discriminator = Discriminator().to(device)

# Loss functions
classification_loss = nn.BCELoss().to(device)
discriminator_loss = CustomBCELoss().to(device)
discriminator_loss1 = nn.BCELoss().to(device)

# Optimizers
optimizer_fe_real = torch.optim.Adam(feature_extractor_real.parameters(), lr=0.001)
optimizer_fe_sim = torch.optim.Adam(feature_extractor_sim.parameters(), lr=0.001)
optimizer_classifier_real = torch.optim.Adam(classifier_real.parameters(), lr=0.001)
optimizer_classifier_sim = torch.optim.Adam(classifier_sim.parameters(), lr=0.001)
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.001)
t = []
ct = []
cd = []
cs = []
cr = []
cb = []
# ***********************************************************************************************
#                              Maximum Mean Discrepancy
# **********************************************************************************************


import sklearn
from sklearn.metrics.pairwise import pairwise_distances



def gaussian_kernel(x, y, sigma=1.0):
    """
    Compute the Gaussian (RBF) kernel between x and y.

    Parameters:
    - x: A tensor of shape (m, features).
    - y: A tensor of shape (n, features).
    - sigma: The bandwidth of the kernel.

    Returns:
    - A (m, n) tensor representing the kernel matrix.
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.view(x_size, 1, dim)
    y = y.view(1, y_size, dim)
    kernel = torch.exp(-torch.sum((x - y) ** 2, dim=2) / (2 * sigma ** 2))
    return kernel

def mmd_loss(x, y, sigma=1.0):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two batches x and y.

    Parameters:
    - x: A tensor of shape (m, features).
    - y: A tensor of shape (n, features).
    - sigma: The bandwidth of the kernel.

    Returns:
    - The MMD loss as a single scalar.
    """
    x_kernel = gaussian_kernel(x, x, sigma)
    y_kernel = gaussian_kernel(y, y, sigma)
    xy_kernel = gaussian_kernel(x, y, sigma)

    return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

def median_heuristic(X, Y):
    # Combine all the data
    combined_data = torch.cat((X, Y), axis=0)
    combined_data = combined_data.detach().cpu().numpy()
    # Compute all pairwise distances
    pairwise_distances = sklearn.metrics.pairwise.euclidean_distances(combined_data, combined_data)
    # Take the median of the upper triangle of the distance matrix, excluding the diagonal
    median_distance = np.median(pairwise_distances[np.triu_indices_from(pairwise_distances, k=1)])
    sigma = median_distance
    return sigma
# Assuming you have dataloaders for real and simulated data: real_dataloader, sim_dataloader
num_epochs = 100
for epoch in range(num_epochs):
    n=0
    accuracy = 0
    mmd_value_before = 0
    mmd_value_after = 0
    loss_sim = 0
    loss_real = 0
    lossdic = 0
    lossdic_own= 0
    losstotal = 0

    for (real_images, real_labels), (sim_images, sim_labels) in zip(real_loader, sim_loader):
        real_labels = real_labels.unsqueeze(1)
        real_labels = real_labels.float()
        sim_labels = sim_labels.unsqueeze(1)
        sim_labels = sim_labels.float()
        real_images = real_images.to(device)
        real_labels = real_labels.to(device)
        sim_images = sim_images.to(device)
        sim_labels = sim_labels.to(device)



        x_flat = real_images.view(real_images.size(0), -1)
        y_flat = sim_images.view(sim_images.size(0), -1)

        # Compute MMD loss

        a = median_heuristic(x_flat, y_flat)
        mmd_value_before = mmd_value_before + mmd_loss(x_flat, y_flat, sigma=a)
        # Zero the gradients for all optimizers
        optimizer_fe_real.zero_grad()
        optimizer_fe_sim.zero_grad()
        optimizer_classifier_real.zero_grad()
        optimizer_classifier_sim.zero_grad()
        optimizer_discriminator.zero_grad()

        # Forward pass for feature extractors and classifiers
        # loss_before =+ mmd_loss(real_images,sim_images)
        real_features = feature_extractor_real(real_images)
        sim_features = feature_extractor_sim(sim_images)

        real_preds = classifier_real(real_features)
        sim_preds = classifier_sim(sim_features)


        # Classification loss for real and simulated data
        loss_classifier_real = classification_loss(real_preds, real_labels)
        loss_classifier_sim = classification_loss(sim_preds, sim_labels)
        total_classifier = loss_classifier_sim + loss_classifier_real


        real_discriminator_out = discriminator(real_features)
        sim_discriminator_out = discriminator(sim_features)

        a = median_heuristic(real_features, sim_features)
        mmd_value_after = mmd_value_after + mmd_loss(real_features, sim_features, sigma=a)

        real_lable = torch.ones_like(real_discriminator_out)
        sim_label = torch.zeros_like(real_discriminator_out)
        real_discriminator_loss1 = discriminator_loss(real_discriminator_out, sim_label)
        sim_discriminator_loss1 = discriminator_loss(sim_discriminator_out, real_lable)
        discriminatorloss = (real_discriminator_loss1 + sim_discriminator_loss1)*1
        # Classification loss for real and simulated data

        loss_total = 1*discriminatorloss + total_classifier + 0.1*(mmd_loss(real_features, sim_features, sigma=a))
        loss_total.backward()
        optimizer_fe_real.step()
        optimizer_fe_sim.step()
        optimizer_classifier_real.step()
        optimizer_classifier_sim.step()
        optimizer_discriminator.step()

        # Update classifiers
        # # loss_classifier_real.backward(retain_graph=True)
        # total_classifier.backward(retain_graph=True)
        # optimizer_classifier_real.step()
        # optimizer_fe_real.step()
        # # loss_classifier_sim.backward(retain_graph=True)
        # optimizer_classifier_sim.step()
        # optimizer_fe_sim.step()
        # optimizer_fe_real.zero_grad()
        # optimizer_fe_sim.zero_grad()
        # optimizer_classifier_real.zero_grad()
        # optimizer_classifier_sim.zero_grad()

        optimizer_discriminator.zero_grad()

        real_discriminator_out = discriminator(real_features.detach())
        sim_discriminator_out = discriminator(sim_features.detach())
        real_lable = torch.ones_like(real_discriminator_out)
        real_discriminator_loss = discriminator_loss1(real_discriminator_out, real_lable)
        sim_label = torch.zeros_like(sim_discriminator_out)
        sim_discriminator_loss = discriminator_loss1(sim_discriminator_out, sim_label)
        total_dec = real_discriminator_loss + sim_discriminator_loss
        total_dec.backward()
        optimizer_discriminator.step()









        accuracy = accuracy + discriminator_accuracy(discriminator, real_features, sim_features)
        loss_real =  loss_real + loss_classifier_real
        loss_sim = loss_sim + loss_classifier_sim
        losstotal = losstotal + loss_total
        lossdic = lossdic + discriminatorloss
        lossdic_own = lossdic_own + total_dec
        n=n+1

    print(f'Epoch {epoch+1}/{num_epochs}, loss Total: {losstotal/n},loss_discriminator: { lossdic/n}, loss_discriminator: {lossdic_own/n} ')

    print(f'Epoch {epoch+1}/{num_epochs}, loss_classifier_real: {loss_real/n},loss_classifier_sim: { loss_sim/n}')

    print(f'Epoch {epoch+1}/{num_epochs}, accuracy:{(accuracy/n)*100}')

    print(f'MMD Value Before: {mmd_value_before.item()/n}, MMD Value After: {mmd_value_after.item()/n}')
    torch.save(feature_extractor_real, 'C:\\Users\\movafagm\\moein\\Saved\\feature_extractor_real.pth')
    torch.save(feature_extractor_sim, 'C:\\Users\\movafagm\\moein\\Saved\\feature_extractor_sim.pth')
    torch.save(classifier_sim, 'C:\\Users\\movafagm\\moein\\Saved\\classifier_sim.pth')
     

    # if loss_total < .01 and  loss_classifier_sim < .01 and loss_classifier_real< .01:
    #     break
    t.append(loss_total)
    ct.append(total_dec)
    cd.append(real_discriminator_loss)
    cb.append(sim_discriminator_loss)
    cs.append(real_discriminator_loss1)
    cr.append(accuracy*100/n)


# t=torch.tensor(t).squeeze()
# ct=torch.tensor(ct).squeeze()
# cd=torch.tensor(cd).squeeze()
# cs=torch.tensor(cs).squeeze()
cr=torch.tensor(cr).squeeze()
# cb=torch.tensor(cb).squeeze()


indices = torch.arange(0, len(t))
plt.figure(figsize=(12, 8))
# plt.plot(indices, t, label='Loss discriminator fool')
# plt.plot(indices, ct, label='Loos discriminator')
# plt.plot(indices, cd, label='Real discriminator')
# plt.plot(indices, cb, label='Sim discriminator')
plt.plot(indices, cr, label='accuracy of discriminator ')
# plt.plot(indices, cs, label='Real discriminator fooling')
plt.xlabel('epoch')
plt.ylabel('Values')
plt.title('Visualization of loss')
plt.legend()
plt.grid(True)
plt.show()


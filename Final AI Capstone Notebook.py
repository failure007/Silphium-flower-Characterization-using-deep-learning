#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[40]:


import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import ParameterSampler
import torchvision.transforms as transforms


# # Data loading and preprocessing

# In[41]:


# Path to CSV and image data
data_path = "C:/Users/DheerajKumarJallipal/Desktop/final data.csv"
image_dir = "C:/Users/DheerajKumarJallipal/Desktop/Siliphium/images"


# In[42]:


# Load and preprocess the CSV data
data = pd.read_csv(data_path)
data['filename'] = data['filename'].str.replace('.jpg', '', regex=False).astype(str)


# In[43]:


# Normalize region count
scaler = StandardScaler()
data['region_count'] = scaler.fit_transform(data['region_count'].values.reshape(-1, 1))


# In[44]:


# Create a lookup dictionary for filename to region count
filename_count_lookup = {}
for i, row in data.iterrows():
    fname = row['filename']
    if 'modified' in fname:
        fname = fname.replace('modified_', '')
    filename_count_lookup[fname] = row['region_count']


# In[45]:


# Read and preprocess images, normalizing pixel values
images = []
counts = []
for filename, count in filename_count_lookup.items():
    img_path = os.path.join(image_dir, f"{filename}.jpg")
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))  # Standardize image size
            images.append(img / 255.0)  # Normalize pixel values
            counts.append(count)


# In[46]:


# Convert to numpy arrays
images = np.array(images)
counts = np.array(counts)


# In[47]:


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, counts, test_size=0.2, random_state=42)


# In[48]:


# Convert to PyTorch tensors with correct shape
X_train_tensor = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32)  # (C, H, W)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32)  # (C, H, W)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)


# In[49]:


# Custom dataset class with transformation support
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)  # Apply transformation
        return x, y


# # Defining a custom CNN model for regression

# In[50]:


class CNNModel(nn.Module):
    def __init__(self, num_filters=32, dropout_rate=0.5):
        super(CNNModel, self).__init__()
        # Convolution layers with pooling
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(num_filters, num_filters*2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(num_filters*2, num_filters*4, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(num_filters*4, num_filters*4, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.flat_dim = num_filters*4 * (224 // 16) * (224 // 16)
        self.fc1 = nn.Linear(self.flat_dim, 512)
        self.fc2 = nn.Linear(512, 1)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Apply convolutional layers with pooling
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)

        # Flatten and fully connected layers
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)

        return x


# In[51]:


# Initialize the model and optimizer
model = CNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # optimal learning rate


# In[52]:


# Define data loaders with augmentation for training and minimal transformation for testing
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


# In[53]:


train_loader = DataLoader(AugmentedDataset(train_dataset, transform=transform_train), batch_size=32, shuffle=True)
test_loader = DataLoader(AugmentedDataset(test_dataset, transform=transform_test), batch_size=32, shuffle=False)


# # Training and Validation with Custom CNN
# 

# In[55]:


# Train the custom CNN model
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    model.train()  # Set to training mode
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()  # Reset gradients
        outputs = model(X_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / len(train_loader)}")


# In[56]:


# Validation with the custom CNN model
model.eval()  # Set to evaluation mode
with torch.no_grad():  # Disable gradient computation for validation
    total_test_loss = 0
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Calculate validation loss
        total_test_loss += loss.item()

print(f"Validation Loss: {total_test_loss / len(test_loader)}")


# # Building and Training ResNet-18

# In[57]:


resnet18 = models.resnet18(pretrained=True)  # Load pre-trained ResNet-18
resnet18.fc = nn.Linear(resnet18.fc.in_features, 1)  # Adjust for regression

optimizer = optim.Adam(resnet18.parameters(), lr=0.001)  # Adjustable learning rate

# Train the ResNet-18 model
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    resnet18.train()  # Set to training mode
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()  # Reset gradients
        outputs = resnet18(X_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / len(train_loader)}")

# Validation with ResNet-18
resnet18.eval()  # Set model to evaluation mode
with torch.no_grad():  
    total_test_loss = 0
    for X_batch, y_batch in test_loader:
        outputs = resnet18(X_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Calculate validation loss
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_loader)  # Average test loss
print(f"Validation Loss with ResNet-18: {avg_test_loss}")


# # Exploring Learning Rates for ResNet-18 Regression

# In[58]:


#Experimenting custom Learning rates for model

# Define a function to conduct a single training run with given hyperparameters
def train_with_hyperparameters(learning_rate, num_epochs, train_loader, test_loader):
    # Load ResNet-18 pre-trained on ImageNet and modify the final layer for regression
    resnet18 = models.resnet18(pretrained=True)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 1)  # Single output for regression
    optimizer = optim.Adam(resnet18.parameters(), lr=learning_rate)  # Adjust learning rate
    criterion = nn.MSELoss()
    
    total_iterations = 0
    train_losses = []
    test_losses = []

    # Train the model
    for epoch in range(num_epochs):
        resnet18.train()  # Set to training mode
        total_epoch_loss = 0  # For average loss calculation
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()  # Reset gradients
            outputs = resnet18(X_batch)  # Forward pass
            loss = criterion(outputs, y_batch)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            train_losses.append(loss.item())
            total_epoch_loss += loss.item()
            total_iterations += 1

        # Average epoch loss
        avg_epoch_loss = total_epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_epoch_loss}")

    # Validation
    resnet18.eval()  # Set model to evaluation mode
    with torch.no_grad():
        total_test_loss = 0
        for X_batch, y_batch in test_loader:
            outputs = resnet18(X_batch)  # Forward pass
            loss = criterion(outputs, y_batch)  # Compute validation loss
            test_losses.append(loss.item())
            total_test_loss += loss.item()

    # Average test loss
    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Validation Loss: {avg_test_loss}")

    return avg_test_loss, train_losses

# Define the ranges for hyperparameters
learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
param_grid = {
    'learning_rate': learning_rates,
    'num_epochs': [10]
}

# Create a plot to visualize loss over iterations
plt.figure(figsize=(10, 6))

# Iterate through different learning rates
for lr in learning_rates:
    print(f"Training with Learning Rate: {lr}")
    avg_test_loss, train_losses = train_with_hyperparameters(lr, 10, train_loader, test_loader)
    
    # Plot the training loss over iterations for each learning rate
    plt.plot(range(len(train_losses)), train_losses, label=f"LR: {lr}, Test Loss: {avg_test_loss:.4f}")

plt.xlabel("Iterations")
plt.ylabel("Training Loss (Log Scale)")
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.title("Training Loss Over Iterations with Different Learning Rates")
plt.legend()
plt.show()


# # Hyperparameter Optimization

# In[59]:


# Define a function to conduct a single training run with given hyperparameters
def train_with_hyperparameters(resnet_model, criterion, optimizer, num_epochs, train_loader, test_loader):
    total_loss = 0
    for epoch in range(num_epochs):
        total_loss = 0
        resnet_model.train()  # Set to training mode
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()  # Reset gradients
            outputs = resnet_model(X_batch)  # Forward pass
            loss = criterion(outputs, y_batch)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_loss += loss.item()

    resnet_model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        total_test_loss = 0
        for X_batch, y_batch in test_loader:
            outputs = resnet_model(X_batch)  # Forward pass
            loss = criterion(outputs, y_batch)  # Calculate validation loss
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)  # Average test loss

    return avg_test_loss

# Define the ranges for hyperparameters
param_grid = {
    'learning_rate': [0.01, 0.005, 0.001, 0.0005, 0.0001],
    'num_epochs': [5, 10, 15]
}

# Generate random combinations of hyperparameters
param_list = list(ParameterSampler(param_grid, n_iter=5, random_state=42))

best_loss = float('inf')
best_params = None

# Iterate through different hyperparameter combinations
for params in param_list:
    learning_rate = params['learning_rate']
    num_epochs = params['num_epochs']

    # Load ResNet-18 pre-trained on ImageNet and modify the final layer for regression
    resnet18 = models.resnet18(pretrained=True)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 1)  # Single output for regression

    criterion = nn.MSELoss()
    optimizer = optim.Adam(resnet18.parameters(), lr=learning_rate)

    avg_test_loss = train_with_hyperparameters(resnet18, criterion, optimizer, num_epochs, train_loader, test_loader)

    print(f"Validation Loss for Learning Rate {learning_rate}, Epochs {num_epochs}: {avg_test_loss}")

    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        best_params = params

print(f"Best hyperparameters: {best_params}, Validation Loss: {best_loss}")


# Plotting the best learning rate graphs based on loss results

# In[60]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# This would represent our collected training losses over iterations for each learning rates
train_losses_0001 = [4.118840849399566, 2.4702807903289794, 2.4411700546741484, 2.4314307808876037,
                     2.419631654024124, 2.456296890974045, 2.4057440876960756, 2.4148765921592714,
                     2.3892575800418854, 2.389414018392563]
train_losses_001 = [4.276243066787719, 2.593796068429947, 2.7526249289512634, 2.6160153567790987,
                    2.537800294160843, 2.602378469705582, 2.476749861240387, 2.7259781122207642,
                    2.6644821166992188, 2.5374913215637207]

# Create unique colors for the plots
colors = list(cm.rainbow(np.linspace(0, 1, 2)))  # Since we're plotting two lines

# Plot for learning rate 0.01
plt.figure(figsize=(8, 5))
plt.plot(range(len(train_losses_001)), train_losses_001, label="Learning Rate 0.01", color=colors[0])
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.title("Training Loss Over Iterations for Learning Rate 0.01")
plt.legend()
plt.show()

# Plot for learning rate 0.0001
plt.figure(figsize=(8, 5))
plt.plot(range(len(train_losses_0001)), train_losses_0001, label="Learning Rate 0.0001", color=colors[1])
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.title("Training Loss Over Iterations for Learning Rate 0.0001")
plt.legend()
plt.show()


# # JUST need to check what is below code

# In[61]:


# Load the pre-trained ResNet-18 model
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 1)  # Adjust for regression

# Define the optimizer with the best learning rate
optimizer = optim.Adam(resnet18.parameters(), lr=0.0005)  # Best learning rate found

# Train the ResNet-18 model
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    resnet18.train()  # Set to training mode
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()  # Reset gradients
        outputs = resnet18(X_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / len(train_loader)}")

# Validation with ResNet-18
resnet18.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient computation for validation
    total_test_loss = 0
    for X_batch, y_batch in test_loader:
        outputs = resnet18(X_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Calculate validation loss
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_loader)  # Average test loss
print(f"Validation Loss with ResNet-18: {avg_test_loss}")


# We got
# 
# Validation Loss with ResNet-18: 2.1896552801132203 when we trained with 0.0001 learning rate 
#     &
#     
# Validation Loss with ResNet-18: 2.188282608985901 when we rained with 0.0005 learning rate

# # Grad CAM

# In[ ]:


# Define Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor):
        self.model.zero_grad()  # Reset gradients
        output = self.model(input_tensor)  # Forward pass

        # Create a one-hot gradient for backward pass
        one_hot = torch.zeros_like(output)  # Zero gradient tensor
        one_hot[0, 0] = 1  # Gradient target for regression

        # Backward pass with one-hot tensor
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients.data
        activations = self.activations.data

        # Calculate weights for Grad-CAM
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activations to get heatmap
        weighted_activations = (activations * weights).sum(dim=1, keepdim=True)

        # Normalize and apply ReLU
        heatmap = nn.functional.relu(weighted_activations)
        heatmap /= heatmap.max()

        # Resize to match input image size
        heatmap = torch.nn.functional.interpolate(heatmap, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)

        return heatmap.squeeze().cpu().numpy()

# Define the target layer for Grad-CAM
target_layer = model.conv4  

# Create Grad-CAM instance
grad_cam = GradCAM(model, target_layer)

# Directory containing all images
image_directory = "C:/Users/DheerajKumarJallipal/Desktop/Siliphium/images"  # image directory
image_files = [f for f in os.listdir(image_directory) if f.endswith((".jpg", ".png"))]

# Define preprocessing for the model
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loop through all images in the directory
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)  # Full path to the image file
    try:
        # Open and process the image
        original_image = Image.open(image_path)
        input_tensor = transform(original_image).unsqueeze(0)  # Add batch dimension

        # Generate Grad-CAM heatmap
        heatmap = grad_cam.generate_cam(input_tensor)

        # Resize heatmap to match the original image size
        heatmap = cv2.resize(heatmap, (original_image.width, original_image.height))

        # Apply colormap and overlay heatmap on the original image
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        superimposed_image = heatmap * 0.4 + np.array(original_image)  # Blend heatmap with original image

        # Display the Grad-CAM heatmap overlayed on the original image
        plt.imshow(superimposed_image)
        plt.axis("off")  # Hide axis ticks
        plt.title(f"Grad-CAM for {image_file}")
        plt.show()

    except Exception as e:
        print(f"Error processing {image_file}: {e}")


# In[6]:


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor):
        self.model.zero_grad()  # Reset gradients
        output = self.model(input_tensor)  # Forward pass

        # Create a one-hot gradient for backward pass
        one_hot = torch.zeros_like(output)  # Zero gradient tensor
        one_hot[0, 0] = 1  # Gradient target for regression

        # Backward pass with one-hot tensor
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients.data
        activations = self.activations.data

        # Calculate weights for Grad-CAM
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activations to get heatmap
        weighted_activations = (activations * weights).sum(dim=1, keepdim=True)

        # Normalize and apply ReLU
        heatmap = F.relu(weighted_activations)
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / heatmap.max()

        # Resize to match input image size
        heatmap = cv2.resize(
            heatmap.squeeze().cpu().numpy(),
            (input_tensor.shape[2], input_tensor.shape[3]),
        )

        return heatmap

# Create Grad-CAM instance
grad_cam = GradCAM(model, target_layer)

# Directory containing all images
image_directory = "C:/Users/DheerajKumarJallipal/Desktop/Siliphium/images"  # image directory
image_files = [f for f in os.listdir(image_directory) if f.endswith((".jpg", ".png"))]

# Define preprocessing for the model
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loop through all images in the directory
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)  # Full path to the image file
    try:
        # Open and process the image
        original_image = Image.open(image_path)
        input_tensor = transform(original_image).unsqueeze(0)  # Add batch dimension

        # Generate Grad-CAM heatmap
        heatmap = grad_cam.generate_cam(input_tensor)

        # Apply colormap and normalize heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255  # Normalize

        # Resize the original image to match the heatmap
        original_image_resized = original_image.resize((224, 224))  # Resize
        original_image_np = np.array(original_image_resized) / 255  # Normalize

        # Blend the heatmap with the original image
        superimposed_image = heatmap * 0.5 + original_image_np  # Blend heatmap with original

        # Display the Grad-CAM heatmap overlayed on the original image
        plt.imshow(superimposed_image)
        plt.axis("off")  # Hide axis ticks
        plt.title(f"Grad-CAM for {image_file}")
        plt.show()

    except Exception as e:
        print(f"Error processing {image_file}: {e}")


# In[ ]:





# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

#Hyperparams
input_dim = 2
num_top_classes = 2
num_mid_classes = 4
num_low_classes = 4
num_samples_per_leaf = 100
sigma2 = 1

learning_rate = 0.01
num_epochs = 100
log_interval = 20
batch_size = 64

class HyperbolicMultinomialRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(HyperbolicMultinomialRegression, self).__init__()
        # Define weights without biases to avoid complications in hyperbolic space
        self.weight = nn.Parameter(torch.Tensor(num_classes, input_dim + 1))
        # self.weight = nn.Linear(input_dim, num_classes, bias=False)
        self.register_buffer('eps', torch.tensor(1e-5))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Initialize weights

    # def lorentz_inner_product(self, x, v):
    #     # Compute the Lorentzian inner product
    #     return -x[:, 0] * v[:, 0] + torch.sum(x[:, 1:] * v[:, 1:], dim=1)

    def forward(self, x):
        # Embed input x into hyperbolic space
        x0 = torch.sqrt(1 + torch.sum(x**2, dim=1, keepdim=True))
        x_hyperbolic = torch.cat([x0, x], dim=1)  # Shape: (batch_size, input_dim + 1)

        # Construct hyperplane normal vectors
        w = self.weight  # Shape: (num_classes, input_dim)
        w0 = w[:, 0:1]
        wE = w[:, 1:]
        wE_norm = torch.sqrt(torch.sum(wE**2, dim=1, keepdim=True))  # Shape: (num_classes, 1)
        hyperplanes = wE / (wE_norm + self.eps) * torch.sqrt(w0**2 + 1) #torch.cat([w0, w], dim=1)  # Shape: (num_classes, input_dim + 1)

        # Compute Lorentz inner products between points and hyperplanes
        # Vectorize computation over batch and classes
        x_expanded = x_hyperbolic.unsqueeze(1)  # Shape: (batch_size, 1, input_dim + 1)
        hyperplanes_expanded = hyperplanes.unsqueeze(0)  # Shape: (1, num_classes, input_dim + 1)
        lorentz_products = -x_expanded[:, :, 0] * hyperplanes_expanded[:, :, 0] + \
                           torch.sum(x_expanded[:, :, 1:] * hyperplanes_expanded[:, :, 1:], dim=2)
        
        # neg_products = torch.clamp(-lorentz_products, min=1 + self.eps)
        # # Compute distances using the corrected formula
        # distances = torch.acosh(neg_products)  # Shape: (batch_size, num_classes)
        
        # # Convert negative distances to logits
        # logits = -distances
        return lorentz_products
    
class EuclideanMultinomialRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EuclideanMultinomialRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        logits = self.linear(x)  # Shape: (batch_size, num_classes)
        return logits

# %%

def generate_deep_hierarchical_data(num_samples_per_leaf=100, input_dim=3, num_top_classes=3, num_mid_classes=2, num_low_classes=2):
    np.random.seed(42)
    X = []
    y = []
    class_label = 0

    # Top level
    for i in range(num_top_classes):
        top_mean = np.random.randn(input_dim) * 10 * (i + 1)
        
        # Middle level
        for j in range(num_mid_classes):
            mid_mean = top_mean + np.random.randn(input_dim) * 5 * (j + 1)
            
            # Leaf level
            for k in range(num_low_classes):
                leaf_mean = mid_mean + np.random.randn(input_dim) * 2 * (k + 1)
                cov = sigma2 * np.eye(input_dim)
                samples = np.random.multivariate_normal(leaf_mean, cov, num_samples_per_leaf)
                labels = np.full(num_samples_per_leaf, class_label)
                X.append(samples)
                y.append(labels)
                class_label += 1  # Unique label for each leaf

    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y

# Generate the hierarchical data

num_labels = num_mid_classes * num_top_classes * num_low_classes

X, y = generate_deep_hierarchical_data(num_samples_per_leaf=num_samples_per_leaf, 
                                       input_dim=input_dim, 
                                       num_top_classes=num_top_classes, 
                                       num_mid_classes=num_mid_classes,
                                       num_low_classes=num_low_classes)

# Visualize with a 3D Scatter Plot
# fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# for label in np.unique(y):
#     class_points = X[y == label]
#     ax.scatter(class_points[:, 0], class_points[:, 1], label=f'Leaf {label+1}', alpha=0.6)
# ax.set_title("3D Scatter Plot of Deep Hierarchical Data")
# ax.set_xlabel("Feature 1")
# ax.set_ylabel("Feature 2")
# ax.legend(bbox_to_anchor=(1.05, 1))
# plt.show()

# %%

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.111)


# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).long()
X_val_tensor = torch.from_numpy(X_test).float()
y_val_tensor = torch.from_numpy(y_test).long()

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, device, X_train, y_train, X_val, y_val):
    model = model.to(device)
    
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    # Create DataLoader
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):

        model.train()
        epoch_loss = 0
        epoch_norm = 0

        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e5)
            epoch_norm += norm
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val)
            val_probs = F.softmax(val_logits, dim=1)
            _, val_predicted = torch.max(val_probs, 1)
            val_accuracy = (val_predicted == y_val).float().mean().item()

        if epoch % log_interval == 0:
            print(f"Epoch {epoch}: Train Loss {epoch_loss/len(dataloader):.2f}, Val Loss {val_loss.item():.2f}, Val Acc {val_accuracy*100:.2f}%, Grad Norm {epoch_norm:.1f}")

    print("Training finished")
    return model



def evaluate_model(model, X_test, y_test):
    model, X_test, y_test = model.to(device), X_test.to(device), y_test.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = F.softmax(logits, dim=1)
        _, predicted = torch.max(probs, 1)
        accuracy = (predicted == y_test).float().mean().item()
    return accuracy

# %%
# Instantiate models
euclidean_model = EuclideanMultinomialRegression(input_dim, num_labels)
hyperbolic_model = HyperbolicMultinomialRegression(input_dim, num_labels)

hyperbolic_model = train_model(hyperbolic_model, device, 
                               X_train_tensor, y_train_tensor, 
                               X_val_tensor, y_val_tensor)

# Train Euclidean model
euclidean_model = train_model(euclidean_model, device, 
                              X_train_tensor, y_train_tensor, 
                              X_val_tensor, y_val_tensor)
# Evaluate models
euclidean_accuracy = evaluate_model(euclidean_model, X_test_tensor, y_test_tensor)
hyperbolic_accuracy = evaluate_model(hyperbolic_model, X_test_tensor, y_test_tensor)

print(f"Euclidean Model Accuracy: {euclidean_accuracy * 100:.2f}%")
print(f"Hyperbolic Model Accuracy: {hyperbolic_accuracy * 100:.2f}%")
# %%




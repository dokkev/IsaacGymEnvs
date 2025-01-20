import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

# Define the dataset class
class RolloutDataset(Dataset):
    def __init__(self, rollouts, history_len):
        self.data = []
        for rollout in rollouts:
            if rollout['done'] == 1:  # End of trial
                continue
            proprio_hist = torch.tensor(rollout['proprio_hist'], dtype=torch.float32)
            priv_info = torch.tensor(rollout['priv_info'], dtype=torch.float32)

            self.data.append((proprio_hist, priv_info))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # RNN forward pass
        _, (hidden, _) = self.rnn(x)
        hidden = hidden[-1]  # Use the last layer's hidden state
        output = self.fc(hidden)
        return output

# Load rollouts
rollouts = torch.load('rollouts.pt')

# Hyperparameters
input_dim = 32
hidden_dim = 64
output_dim = rollouts[0]['priv_info'].shape[0]
num_layers = 2
history_len = 30
batch_size = 16
epochs = 100
learning_rate = 0.001

# Prepare dataset
train_dataset = RolloutDataset(rollouts[:int(0.8 * len(rollouts))], history_len)
test_dataset = RolloutDataset(rollouts[int(0.8 * len(rollouts)):], history_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for proprio_hist, priv_info in train_loader:
        # Move data to the device
        proprio_hist = proprio_hist.to(device)
        priv_info = priv_info.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(proprio_hist)
        loss = criterion(output, priv_info)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")



# Evaluate the model
def evaluate_model():
    model.eval()
    predictions, ground_truths = [], []
    with torch.no_grad():
        for proprio_hist, priv_info in test_loader:
            proprio_hist = proprio_hist.to(device)
            priv_info = priv_info.to(device)
            output = model(proprio_hist)
            
            # Move tensors to CPU before converting to NumPy
            predictions.append(output.cpu().numpy())
            ground_truths.append(priv_info.cpu().numpy())

    predictions = np.vstack(predictions)
    ground_truths = np.vstack(ground_truths)
    return predictions, ground_truths

predictions, ground_truths = evaluate_model()


# Plot training loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

# Scatter plot of predictions vs ground truth
plt.subplot(1, 2, 2)
plt.scatter(ground_truths.flatten(), predictions.flatten(), alpha=0.5, label='Predictions')
plt.scatter(ground_truths.flatten(), ground_truths.flatten(), alpha=0.7, label='Ground Truth', color='red', marker='x')
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.title('Predictions vs Ground Truth')
plt.legend()

plt.tight_layout()
plt.show()

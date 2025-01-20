import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1) Dataset class (same as before)
# ------------------------------------------------------
class RolloutDataset(Dataset):
    def __init__(self, rollouts, history_len):
        self.data = []
        for rollout in rollouts:
            if rollout['done'] == 1:
                continue
            proprio_hist = torch.tensor(rollout['proprio_hist'], dtype=torch.float32)

            priv_info = torch.tensor(rollout['priv_info'])

            # Convert from one-hot to integer label if needed
            if priv_info.dim() == 1 and priv_info.shape[0] > 1:
                priv_info = priv_info.argmax(dim=0)

            priv_info = priv_info.long().squeeze()
            self.data.append((proprio_hist, priv_info))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ------------------------------------------------------
# 2) RNN Model
# ------------------------------------------------------
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        _, (hidden, _) = self.rnn(x)
        last_hidden = hidden[-1]  # [batch_size, hidden_dim]
        logits = self.fc(last_hidden)  # [batch_size, num_classes]
        return logits


# ------------------------------------------------------
# 3) Load rollouts and create train/test sets
# ------------------------------------------------------
rollouts = torch.load('rollouts.pt')

# Hyperparameters
input_dim   = 32
hidden_dim  = 64
num_classes = 5   # Example: 5 classes
num_layers  = 2
history_len = 30
batch_size  = 16
epochs      = 20
learning_rate = 1e-3

# Split
split_idx = int(0.8 * len(rollouts))
train_rollouts = rollouts[:split_idx]
test_rollouts  = rollouts[split_idx:]

train_dataset = RolloutDataset(train_rollouts, history_len)
test_dataset  = RolloutDataset(test_rollouts, history_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------------------------------
# 4) Initialize Model, Loss, Optimizer
# ------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNModel(input_dim, hidden_dim, num_classes, num_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ------------------------------------------------------
# 5) Evaluation Function (returns loss & accuracy)
# ------------------------------------------------------
def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for proprio_hist, priv_info in data_loader:
            proprio_hist = proprio_hist.to(device)
            priv_info = priv_info.to(device)

            logits = model(proprio_hist)
            loss = criterion(logits, priv_info)
            total_loss += loss.item() * proprio_hist.size(0)

            _, predicted = torch.max(logits, dim=1)
            total_correct += (predicted == priv_info).sum().item()
            total_samples += priv_info.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

# ------------------------------------------------------
# 6) Training Loop (with test set evaluation each epoch)
# ------------------------------------------------------
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_samples = 0

    for proprio_hist, priv_info in train_loader:
        proprio_hist = proprio_hist.to(device)
        priv_info = priv_info.to(device)

        optimizer.zero_grad()
        logits = model(proprio_hist)
        loss = criterion(logits, priv_info)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * proprio_hist.size(0)
        _, predicted = torch.max(logits, dim=1)
        running_correct += (predicted == priv_info).sum().item()
        running_samples += priv_info.size(0)

    # Compute average training metrics
    train_loss = running_loss / running_samples
    train_acc = running_correct / running_samples
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Evaluate on test set
    test_loss, test_acc = evaluate_model(model, test_loader)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

# ------------------------------------------------------
# 7) Plot the results
# ------------------------------------------------------
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(12, 5))

# (a) Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
plt.plot(epochs_range, test_losses, label='Test Loss', marker='x')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# (b) Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Train Acc', marker='o')
plt.plot(epochs_range, test_accuracies, label='Test Acc', marker='x')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0,1.4])
plt.legend()

plt.tight_layout()
plt.show()

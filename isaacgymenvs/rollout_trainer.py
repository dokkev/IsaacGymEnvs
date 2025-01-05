import torch
import argparse
import torch.nn as nn
import torch.optim as optim

def load_rollouts(file_path):
    """
    Load rollouts from a .pt file.

    Args:
        file_path (str): Path to the .pt file containing rollouts.

    Returns:
        list: A list of rollouts (dictionaries) loaded from the file.
    """
    try:
        rollouts = torch.load(file_path)
        print(f"[INFO] Successfully loaded rollouts from {file_path}.")
        
        if not rollouts:
            print("[INFO] No rollouts to summarize.")
            return
        
        print(f"[INFO] Total rollouts: {len(rollouts)}")

        env_counts = {}
        for transition in rollouts:
            env_id = transition['env_id']
            env_counts[env_id] = env_counts.get(env_id, 0) + 1

        print("[INFO] Transitions per environment:")
        for env_id, count in env_counts.items():
            print(f"  Environment {env_id}: {count} transitions")
        
        return rollouts
    
    except Exception as e:
        print(f"[ERROR] Failed to load rollouts: {e}")
        return None


# Model Definition
class StateActionPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=7):
        """
        Neural network to predict the first 7 elements of the next state.
        """
        super(StateActionPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# Dataset Preparation
def prepare_dataset(rollouts):
    """
    Prepare a dataset for supervised learning from rollouts, predicting only the first 7 elements of next_state.

    Args:
        rollouts (list): A list of rollout dictionaries containing PyTorch tensors.

    Returns:
        tuple: Training and testing data (X_train, X_test, y_train, y_test) as PyTorch tensors.
    """
    states = []
    actions = []
    next_states = []

    for transition in rollouts:
        states.append(transition['state'])  # Current state tensor
        actions.append(transition['action'])  # Action tensor
        next_states.append(transition['next_state'][:7])  # Cube POS + QUAT tensor

    # Combine states and actions as input
    X = torch.cat([torch.stack(states), torch.stack(actions)], dim=1)
    y = torch.stack(next_states)

    # Split into train and test datasets
    train_size = int(0.8 * X.size(0))
    indices = torch.randperm(X.size(0))  # Shuffle indices
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


# Training Function
def train_model(model, X_train, y_train, epochs=100, batch_size=32, learning_rate=0.001):
    """
    Train the neural network model.

    Args:
        model (nn.Module): The PyTorch model to train.
        X_train (torch.Tensor): Training input data.
        y_train (torch.Tensor): Training output data.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to device
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")


# Evaluation Function
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data.

    Args:
        model (nn.Module): The trained PyTorch model.
        X_test (torch.Tensor): Test input data.
        y_test (torch.Tensor): Test output data.

    Returns:
        float: Mean squared error on the test set.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Ensure the model is on the device
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = nn.MSELoss()(predictions, y_test)
        print(f"Mean Squared Error on Test Set: {mse.item():.4f}")


def save_model(model, file_path):
    """
    Save the trained model to a file.

    Args:
        model (nn.Module): The trained PyTorch model.
        file_path (str): Path to save the model.
    """
    torch.save(model.state_dict(), file_path)
    print(f"[INFO] Model saved to {file_path}.")


# Main Script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a State-Action Predictor model.")
    parser.add_argument("--rollouts", type=str, required=True, help="Path to the rollouts file (.pt).")
    parser.add_argument("--save_model", type=str, default="state_action_predictor.pth", help="Path to save the trained model.")
    parser.add_argument("--load_model", type=str, help="Path to load a pretrained model.")
    args = parser.parse_args()

    # Load rollouts
    rollouts = load_rollouts(args.rollouts)
    if rollouts is None:
        exit(1)

    # Prepare dataset
    X_train, X_test, y_train, y_test = prepare_dataset(rollouts)

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)

    # Define model
    input_dim = X_train.size(1)
    output_dim = 7
    model = StateActionPredictor(input_dim, output_dim).to(device)

    # if a pretrained model is provided, load it
    if args.load_model:
        model = load_model(model, args.load_model)
    else:
        print("[INFO] No pretrained model provided. Training from scratch...")
        # Train model
        print("[INFO] Training model...")
        train_model(model, X_train, y_train, epochs=100)

    # Save model
    save_model(model, args.save_model)

    # Evaluate model
    print("[INFO] Evaluating model...")
    evaluate_model(model, X_test, y_test)


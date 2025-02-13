import torch
import torch.nn as nn
import torch.optim as optim
import dgl
from dgl.data import AmazonCoBuyComputerDataset
import wandb
import numpy as np
from sklearn.metrics import accuracy_score

# Initialize Weights & Biases
wandb.init(project="mlp_hyperparam_sweep", entity="your_wandb_username")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        super(MLP, self).__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def load_data():
    dataset = AmazonCoBuyComputerDataset()
    graph = dataset[0]
    features = graph.ndata['feat']
    labels = graph.ndata['label']
    
    # Train/val/test split (default from DGL)
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    
    return features, labels, train_mask, val_mask, test_mask

def train_and_evaluate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, labels, train_mask, val_mask, test_mask = load_data()

    model = MLP(
        input_dim=features.shape[1],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        output_dim=len(labels.unique())
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    features, labels = features.to(device), labels.to(device)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    best_val_acc = 0
    best_model = None

    for epoch in range(config["epochs"]):
        model.train()
        logits = model(features)
        loss = criterion(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features)[val_mask]
            val_preds = val_logits.argmax(dim=1)
            val_acc = accuracy_score(labels[val_mask].cpu(), val_preds.cpu())

        # Log metrics
        wandb.log({"epoch": epoch, "val_acc": val_acc, "loss": loss.item()})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()

    # Test final model
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        test_logits = model(features)[test_mask]
        test_preds = test_logits.argmax(dim=1)
        test_acc = accuracy_score(labels[test_mask].cpu(), test_preds.cpu())

    return test_acc

sweep_config = {
    "method": "bayes",  # Bayesian optimization
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 1e-4, "max": 1e-2},
        "hidden_dim": {"values": [64, 128, 256, 512]},
        "num_layers": {"values": [2, 3, 4, 5]},
        "dropout": {"min": 0.2, "max": 0.7},
        "weight_decay": {"min": 1e-5, "max": 1e-2},
        "epochs": {"value": 500}  # Fixed, as long training is needed
    }
}

def sweep_train():
    wandb.init()
    config = wandb.config
    test_acc = train_and_evaluate(config)
    wandb.log({"test_acc": test_acc})

sweep_id = wandb.sweep(sweep_config, project="mlp_hyperparam_sweep")
wandb.agent(sweep_id, function=sweep_train, count=50)  # Run 50 sweeps

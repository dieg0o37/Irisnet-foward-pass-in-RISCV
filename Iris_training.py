import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from typing import List, Tuple


# Define the Iris model
class irismodel(torch.nn.Module):
    def __init__(self, layers: List[Tuple[int, int]]):
        super(irismodel, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(in_features, out_features) for in_features, out_features in layers])
        self.activations = torch.nn.ModuleList([torch.nn.ReLU() for _ in layers])
        self.drop = torch.nn.Dropout(0.2)

    def forward(self, x):
        for i, layer in enumerate(self.linears):
            x = layer(x)
            x = self.activations[i](x)
        x = self.drop(x)
        return x


# Training the model
def fit(model, data, targets, epochs = 400, lr = 0.002):
    print('Training Model...')
    criterio = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = [] # For plotting later
    for _ in range(epochs):
        loss = criterio(model(data), targets)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses



# Testing the model
def test(model, data, targets):
    print("Testing...")
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        preds = outputs.argmax(dim=1)
        acc = accuracy_score(targets.cpu(), preds.cpu())
        cm = confusion_matrix(targets.cpu(), preds.cpu())
    model.train()
    return acc, cm

# Extracting weights
def extract_weights(model):
    print("extracting weights...")
    weights = {}
    for i, layer in enumerate(model.linears):
        weights[f"l{i + 1}"] = layer.weight.data.cpu().numpy().tolist()
    return weights

if __name__ == "__main__":
    iris = load_iris()
    data_train, data_test, answers_train, answers_test = train_test_split(iris.data, iris.target, test_size=0.2)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data_tr_tensor = torch.tensor(data_train, dtype=torch.float32).to(device)
    answers_tr_tensor = torch.tensor(answers_train, dtype=torch.long).to(device)
    data_te_tensor = torch.tensor(data_test, dtype=torch.float32).to(device)
    answers_te_tensor = torch.tensor(answers_test, dtype=torch.long).to(device)

    model = irismodel([(4, 10), (10, 20), (20, 3)]).to(device)
    loss_arr = fit(model, data_tr_tensor, answers_tr_tensor, epochs=400, lr=0.002)


    acc, cm = test(model, data_te_tensor, answers_te_tensor)
    print(f"Test Accuracy: {acc}")
    print(f"Confusion Matrix:\n{cm}")



# plt.plot(loss_arr)
# plt.title("Training curve")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.show()

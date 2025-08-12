from sklearn.metrics import confusion_matrix, accuracy_score
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import Iris_training as ir
import numpy as np

def get_nn_input():
    while True:
        try:
            nn_structure_input = [int(x) for x in input("Enter NN structure (ex: 4,8,15,3): ").strip().split(',')]
            if len(nn_structure_input) < 2:
                print("Structure must have at least input and output layers.")
                continue
            if nn_structure_input[-1] != 3:
                print("Output layer must have 3 neurons.")
                continue
            number_epochs = int(input("Enter number of epochs: "))
            if number_epochs <= 0:
                print("Epochs must be positive.")
                continue
            learning_rate = float(input("Enter learning rate: "))
            if not (0 < learning_rate < 1):
                print("Learning rate should be between 0 and 1.")
                continue
            return nn_structure_input, number_epochs, learning_rate
        except Exception as e:
            print(f"Invalid input: {e}. Please try again.")

nn_structure_input, number_epochs, learning_rate = get_nn_input()

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

iris = load_iris()
data_train, data_test, answers_train, answers_test = train_test_split(iris.data, iris.target, test_size=0.2)
data_tr_tensor = torch.tensor(data_train, dtype=torch.float32).to(device)
answers_tr_tensor = torch.tensor(answers_train, dtype=torch.long).to(device)
data_te_tensor = torch.tensor(data_test, dtype=torch.float32).to(device)
answers_te_tensor = torch.tensor(answers_test, dtype=torch.long).to(device)

nn_structure = []

for x in range(len(nn_structure_input) - 1):
    nn_structure.append((nn_structure_input[x], nn_structure_input[x + 1]))

model = ir.irismodel(nn_structure).to(device)
loss_arr = ir.fit(model, data_tr_tensor, answers_tr_tensor, epochs=number_epochs, lr=learning_rate)

weights = ir.extract_weights(model)
def get_quantized_weights(weights):
    quantized = {}
    for layer, params in weights.items():
        quantized[layer] = [[max(-128, min(127, int(round(p * 127)))) for p in row] for row in params]
    return quantized

def convert_weights_to_string(quantized_weights):
    weight_string = "{"
    for layer, params in quantized_weights.items():
        weight_string += f'"{layer}": ['
        for row in params:
            weight_string += "[" + ",".join(map(str, row)) + "],"
        weight_string = weight_string[:-1] + "],"
    weight_string = weight_string[:-1] + "}"
    return weight_string

weight_string = convert_weights_to_string(get_quantized_weights(weights))
# print(weight_string)
RV32I_test_cases = f" --- NN_structure --- \n"
for i in range(len(nn_structure_input) - 1):
    RV32I_test_cases += str(nn_structure_input[i]) + ","
RV32I_test_cases += str(nn_structure_input[-1]) + f"\n"
RV32I_test_cases += f"--- Weight Matrix --- \n"
RV32I_test_cases += weight_string + f"\n"


def get_quantized_activation_vector(activation_vector):
    return [max(-128, min(127, int(round(p * 127)))) for p in activation_vector.cpu().numpy()]

def get_activation_vector_string(activation_vector):
    return ",".join(map(str, activation_vector))


for i, activation_vector in enumerate(data_te_tensor):
    RV32I_test_cases += f"--- Activation Vector {i} --- \n"
    RV32I_test_cases += get_activation_vector_string(get_quantized_activation_vector(activation_vector)) + f"\n"

# Testing the model
def test(model, data, targets):
    print(f"\n--- Testing with Pytorch ---")
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        preds = outputs.argmax(dim=1)
        acc = accuracy_score(targets.cpu(), preds.cpu())
        cm = confusion_matrix(targets.cpu(), preds.cpu())
    model.train()
    return acc, cm, preds.cpu()

accuracy, conf_matrix, preds = test(model, data_te_tensor, answers_te_tensor)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Confusion Matrix:\n{conf_matrix}")

def create_riscv_test_cases(test_input_string):
    with open("riscv_test_cases.txt", "w") as f:
        f.write(test_input_string)

create_riscv_test_cases(RV32I_test_cases)
   
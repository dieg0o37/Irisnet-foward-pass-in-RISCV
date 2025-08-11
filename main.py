import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import Iris_training as ir
import riscemu

# INPUT FOR PYTHON CODE: 
# 1 Line: NN structure (ex: 4,8,15,3)
# 2 Line: number of epochs
# 3 Line: learning rate

# TRAINING:
# Train Iris model based on INPUT

# EXTRACTING RESULT:
# extract the weight matrizes (no implementation for biases :( )
# transform into ASCII (string) following the format:
# {"l1":[[...]],"l2":[[...]],"l3":[[...]]}

# QUANTIZATION: (RV32I only accepts integer values of 32 bits)
#   solution: transform all float to 8 bit integers [-128, 127] (affine quantization)

# TESTING:
# Get the activation vector for all test cases
# Test the model utilizing Risc-V code

# RISC-V code expected input (ALL ASCII):
# Line 1: 4,8,15,3
# Line 2: {"l1": [[-72, 6, 127, 117], [89, 115, -128, -79], [-83, -56, 127, -54], [-48, -128, 104, 98], [78, 57, -30, -128], [-41, -59, 47, 127], [-128, -29, 36, -45], [-128, 0, -61, 18]], "l2": [[17, -59, 39, 54, -67, 127, -37, -39], [0, -110, 122, -87, 44, -128, 79, -124], [57, -8, -82, 61, 127, -119, -16, -36], [-58, -106, 91, -61, 19, -54, -35, -128], [-36, -88, -128, -83, -5, -57, -88, 116], [-99, 0, 118, -101, -128, -62, 30, 119], [91, -40, -123, -5, 127, 122, -48, 77], [82, 2, 127, -101, -108, -14, 4, -32], [-6, 53, 65, -64, -76, 127, -84, 100], [-57, -101, -97, -56, 72, 5, 127, 60], [-30, 127, 56, -93, 14, -84, 33, 42], [116, 1, 29, 127, 11, -16, 113, 80], [-128, 120, 69, -6, -101, 56, -41, -76], [120, -41, -79, -127, -102, 43, -89, 30], [-29, -53, 127, -57, 45, -8, -105, 44]], "l3": [[7, 8, -107, 0, -45, -24, -127, -17, -19, 12, 82, -98, 63, -39, 14], [-79, 1, 48, 58, -51, -42, 78, -62, -71, 47, 127, 11, -110, 80, 15], [50, -51, 48, -30, -65, 84, 124, 64, 57, -10, -128, 49, 50, 29, -49]]}
# Line 3: 59,30,51,18 (activation vector)

# 1 - Receive INPUT
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
RV32I_input = ""

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
        quantized[layer] = [[int(round(p * 127)) for p in row] for row in params]
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
print(weight_string)
RV32I_input += weight_string + "\n"

def model_testing():
    pass
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import Iris_training as ir
import numpy as np
import subprocess
import os
def risc_v_test_model(test_input_string, pred, correct):
    """
    Runs the irisnet.s file with the provided string as input
    using the riscemu emulator.
    """
    print(f"\n--- Testing with RISC-V Emulator ---")
    # Command to execute the emulator with your assembly file.
    # This assumes 'riscemu' is installed and accessible in your environment.
    command = ["riscemu", "irisnet.s"]

    # Check if the assembly file exists
    if not os.path.exists("irisnet.s"):
        print("Error: 'irisnet.s' not found.")
        return

    try:
        # Run the emulator as a subprocess
        # - 'input' passes the string to the program's standard input.
        # - 'capture_output=True' captures what the program prints.
        # - 'text=True' ensures input and output are treated as strings.
        # - 'check=True' will raise an error if the emulator fails.
        completed_process = subprocess.run(
            command,
            input=test_input_string,
            capture_output=True,
            text=True,
            check=True
        )

        riscv_output = completed_process.stdout.strip()
        print(f"RISC-V program output: {riscv_output}")
        print(f"Expected output: {pred}, Correct answer: {correct}")
        print(f"Test successful!")

    except FileNotFoundError:
        print("Error: 'riscemu' command not found.")
        print("Please ensure you have installed it (e.g., 'pip install riscemu').")
    except subprocess.CalledProcessError as e:
        # If the emulator returns an error, print its error output
        print(f"An error occurred while running the emulator.")
        print(f"Return code: {e.returncode}")
        print(f"Stderr:\\n{e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

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

def quantize_value(x, alfa, beta):
    """
    Aplica a quantização afim para converter um valor float para um inteiro de 8 bits.
    Mapeia o intervalo [min_val, max_val] para o intervalo de inteiros [-128, 127].
    """
    if beta == alfa:
        return 0
    # Calcula a escala e o ponto zero
    scale = (beta - alfa) / 255.0
    zero_point = round(-alfa / scale) - 128 if scale != 0 else 0
    # Quantiza o valor
    quantized_val = int(round(x / scale) + zero_point) if scale != 0 else 0
    # "Clip" (satura) o valor para garantir que ele esteja no intervalo [-128, 127]
    return max(-128, min(127, quantized_val))

def get_clipping_range_weights(weights):
    all_weights = []
    for layer in weights.values():
        for param in layer:
            all_weights.extend(param)
    return (min(all_weights), max(all_weights))

def get_quantized_weights(weights):
    alfa, beta = get_clipping_range_weights(weights)
    quantized_weights = {}
    for layer, params in weights.items():
        quantized_params = [
            [quantize_value(w, alfa, beta) for w in row]
            for row in params
        ]
        quantized_weights[layer] = quantized_params
    return quantized_weights

def convert_weights_to_string(quantized_weights):
    """Converte os pesos quantizados para o formato de string do caso de teste."""
    layer_strings = []
    for layer, params in quantized_weights.items():
        row_strings = [f"[{','.join(map(str, row))}]" for row in params]
        layer_strings.append(f'"{layer}":[{",".join(row_strings)}]')
    return "{" + ",".join(layer_strings) + "}"

def get_clipping_range_ac_vectors(activation_vectors):
    all_activations = activation_vectors.flatten().tolist()
    return (min(all_activations), max(all_activations))

def get_quantized_activation_vector(activation_vector):
    alfa, beta = get_clipping_range_ac_vectors(data_te_tensor)
    return [quantize_value(v.item(), alfa, beta) for v in activation_vector]

def get_activation_vector_string(activation_vector):
    return ",".join(map(str, activation_vector))

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

def create_riscv_test_cases(test_input_string):
    with open("riscv_test_cases.txt", "w") as f:
        print(test_input_string, file=f)


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

weight_string = convert_weights_to_string(get_quantized_weights(weights))
RV32I_test_cases = ""
for i in range(len(nn_structure_input) - 1):
     RV32I_test_cases += str(nn_structure_input[i]) + ","
RV32I_test_cases += str(nn_structure_input[-1]) + "\n"
RV32I_test_cases += weight_string + "\n"

accuracy, conf_matrix, preds = test(model, data_te_tensor, answers_te_tensor)

for i, activation_vector in enumerate(data_te_tensor):
    activation_vec = ""
    activation_vec += get_activation_vector_string(get_quantized_activation_vector(activation_vector))
    activation_vec += "\0"
    with open("riscv_test_cases.txt", "a") as f:
        print(RV32I_test_cases + activation_vec, file=f)
    risc_v_test_model(RV32I_test_cases + activation_vec, preds[i], answers_te_tensor[i].item())


print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Confusion Matrix:\n{conf_matrix}")




   
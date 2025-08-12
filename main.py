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
            if nn_structure_input[-1] != 3 or nn_structure_input[0] != 4:
                print("Output layer must have 3 neurons and input layer must have 4 neurons.")
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
    Applies affine quantization to convert a float value to an 8-bit integer.
    Maps the range [min_val, max_val] to the integer range [-128, 127].
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
    print("Testing complete!")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")
    return preds.cpu()

if __name__ == "__main__":
    nn_structure_input, number_epochs, learning_rate = get_nn_input()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load Iris dataset
    iris = load_iris()
    data_train, data_test, answers_train, answers_test = train_test_split(iris.data, iris.target, test_size=0.2)
    data_tr_tensor = torch.tensor(data_train, dtype=torch.float32).to(device)
    answers_tr_tensor = torch.tensor(answers_train, dtype=torch.long).to(device)
    data_te_tensor = torch.tensor(data_test, dtype=torch.float32).to(device)
    answers_te_tensor = torch.tensor(answers_test, dtype=torch.long).to(device)

    # Define the neural network structure
    nn_structure = []
    for x in range(len(nn_structure_input) - 1):
        nn_structure.append((nn_structure_input[x], nn_structure_input[x + 1]))

    # Create the model based on the defined structure
    model = ir.irismodel(nn_structure).to(device)
    loss_arr = ir.fit(model, data_tr_tensor, answers_tr_tensor, epochs=number_epochs, lr=learning_rate)
    weights = ir.extract_weights(model)

    # Create the input for the RV32I test cases
    weight_string = convert_weights_to_string(get_quantized_weights(weights))
    RV32I_test_cases = ""
    for i in range(len(nn_structure_input) - 1):
        RV32I_test_cases += str(nn_structure_input[i]) + ","
    RV32I_test_cases += str(nn_structure_input[-1]) + "\r\n"
    RV32I_test_cases += weight_string + "\r\n"

    preds = test(model, data_te_tensor, answers_te_tensor)

    # Manual testing with the RISC-V emulator
    with open("riscv_test_cases.txt", "w") as file:
        file.write(f"Load the irisnet.s file into https://riscv-programming.org/ale/#\n")
        file.write(f"Select RUN and copy and paste the test cases below into the terminal:\n")
        pass
    for i, activation_vector in enumerate(data_te_tensor):
        activation_vec = ""
        activation_vec += get_activation_vector_string(get_quantized_activation_vector(activation_vector))
        activation_vec += "\0"
        with open("riscv_test_cases.txt", "a") as f:
            f.write(f"--- Test Case {i} ---\n")
            f.write(f"Expected output: {preds[i]}, Correct answer: {answers_te_tensor[i].item()}\n")
            f.write(f"RISC-V quantized input:\n" + RV32I_test_cases + activation_vec + "\n")
    




    
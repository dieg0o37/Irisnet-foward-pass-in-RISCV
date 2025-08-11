# Iris em Assembly – Redes Neurais com RISC-V
**Feito por: Diego Martins Santos
RA: 288809**

# Descrição do projeto
Esse projeto consistiu em implementar em Assembly RISC-V o processo de inferência de uma rede neural, uma dentre quatro possíveis variações da **IrisNet**, baseada no conjunto de dados Iris. Ou seja, detectar corretamente o tipo da planta Iris baseados nos dados fornecidos pelo terminal.

Cada caso teste possuí 3 linhas, cada uma contendo uma String em ASCII: 
 - a primeira, o tamanho de cada camada de neurônios, 
 - a segunda, as matrizes com os pesos de cada camada da rede
 - a última, os valores para o vetor de ativação inicial da rede. 

Todos os valores dados estão relacionados às medidas das  plantas e foram dados em milímetros. *

O objetivo desse projeto era passar nos casos testes através da implementação do processo de inferência (ou _forward pass_) da IA (Basicamente multiplicação de matrizes por vetores).

> *Obs: Em uma rede neural, é comum que os números sejam representados como ponto flutuante. No entanto, para este trabalho, o modelo passou por um processo de quantização, no qual os números foram convertidos para inteiros de 8 bits. Por isso todos os pesos são valores no intervalo [-127, 127].
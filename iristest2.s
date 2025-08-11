.bss
INPUT_BUFFER: .skip 16384        # Buffer para armazenar a string de entrada inteira
VETOR_ATIVACAO_1: .skip 400     # Buffer para armazenar o vetor de ativação c
VETOR_FINAL: .skip 12           # Buffer para armazenar o vetor final (3 tipos de plantas)
OUTPUT: .skip 4                 # Buffer para armazenar o index da planta escolhida (0, 1 ou 2)
OUTPUT_BUFFER: .skip 2          # Buffer para armazenar o resultado final (0, 1 ou 2) + '\n'
.data
TAM_CAMADAS: .word 4,30,20,10,3 
PESOS_MATRIZ: .word -41, -127, 125, 44, -22, -3, 21, 127, -61, -81, 127, 11, -128, -19, 65, 20, -9, -11, 14, 127, 38, -114, 60, 127, 24, 55, -7, -128, -45, -128, 120, 82, -19, -126, 71, 127, -8, -127, 78, 62, -128, -56, 121, -41, 122, 49, -127, -102, 56, 87, -66, -128, 29, -82, 81, -128, 78, 102, -99, -128, 38, 118, -16, -128, -34, -59, -127, 35, -57, -128, 126, 95, 20, -101, 127, -72, -128, -80, 126, -23, 48, -128, 33, 61, -39, -128, 99, -46, 112, -25, -127, -127, -69, -22, 42, -128, -15, -11, 83, -128, 127, 49, -104, -128, -128, 66, 77, 34, 118, -119, -40, -128, -72, -18, 33, 127, -35, 44, -128, -56, 81, 31, 127, 14, 68, -111, -93, -26, 58, 112, -95, -87, 23, 40, 74, 1, -114, -19, 74, -95, -32, 19, 113, -64, 50, -110, -20, 48, -104, 81, -74, -38, -85, -65, 6, -40, 62, -18, -3, -40, -9, 84, 66, 5, 127, 62, -46, -55, -63, 57, -26, 33, -21, -8, -30, 0, 36, -15, 42, -28, -98, 53, -33, -67, 37, -18, 57, 28, 10, -60, 41, -53, 31, -40, 127, -58, -89, 43, 42, -69, -101, 57, -69, 66, 31, -33, -87, -44, 42, -21, -15, -11, -31, -31, -24, -41, 88, 10, -34, -5, -24, 63, 80, 53, 98, 89, -43, -39, 35, -2, -2, 47, 50, 15, 52, 22, -14, 127, 30, 46, -84, -10, 26, -54, 15, -28, -74, 35, 52, -1, 88, 103, 8, 19, 75, -101, 8, 56, -24, 75, -111, 39, 127, 114, 67, 46, -19, -38, -127, -35, 99, 62, 76, 46, 84, -40, -12, -11, -32, 119, 127, 102, -63, 43, -106, -31, 71, 19, -24, -57, -58, -49, 120, 99, -34, 16, 23, 99, 70, 33, -63, 20, -79, -33, -54, 55, 75, -5, 28, 33, 127, 3, -108, 109, -48, -76, -27, 41, -107, -48, 45, -37, -47, -15, 5, -54, -62, -44, 3, 34, -76, 71, 47, -88, -79, 71, 13, -114, 116, -127, -91, -63, -41, -93, -87, 18, 55, -120, -128, -27, 105, 15, 15, 89, 35, 66, -77, 10, 58, 66, -27, -18, -98, 24, 3, 7, -97, 88, 6, -27, -49, 125, -55, -114, -23, -91, 62, 123, 90, -89, 127, 27, 10, -25, -16, -19, -87, -118, 88, 34, 46, -33, 83, -81, -110, 67, -81, -83, -4, 64, 16, -81, 29, -88, 19, -52, 104, -50, -27, 12, 59, 111, 127, -112, -15, -2, -31, -106, 112, 94, -111, -74, -34, -53, 58, 114, -5, -128, -51, 16, -12, 88, 11, 80, 9, 7, -76, -77, 112, -16, -6, 18, -94, 30, 114, 63, -8, 15, 63, -9, -12, -101, -7, -57, 88, -25, 44, 109, 124, -22, -52, 12, 116, -112, -95, -120, 38, -128, -39, 8, -93, 107, -68, 97, 122, 5, -48, -105, -13, -54, 67, 28, 66, -72, -48, -1, 50, -58, -31, -109, -34, -90, -70, 1, 98, 104, 82, 48, -85, -81, 53, -21, 52, -63, 75, -94, 26, -127, 99, 82, -94, -27, -59, 26, 35, 108, 103, -42, 53, -69, 96, 60, 97, -67, -114, -127, -102, 56, -112, 86, -83, 92, 48, -31, -57, -54, -102, 102, -44, 15, -23, -31, -65, -39, -11, -12, 47, -78, -39, -65, -29, 74, 69, -29, 127, 14, 20, -72, -26, 33, -42, -6, -13, -8, -22, 52, 34, 24, -10, 1, -90, 103, -4, -102, -14, 67, -54, -108, 30, -120, 72, 111, -42, -93, -84, 100, 56, -81, 41, -83, 127, 99, 6, -24, -102, 105, -81, 113, 95, 85, 113, 94, 69, -59, -20, -6, -2, 90, 76, 41, -39, 103, 62, -45, -58, -39, 29, 69, -101, 64, 85, -53, 127, 12, 26, -52, 41, 3, 102, -108, 39, 60, -109, -56, 6, 48, -26, -73, 66, 60, 93, -68, 90, -96, 37, -101, 105, -73, 33, -103, 76, -128, -118, -12, -45, -23, 64, -125, -105, 81, 102, -106, 43, 36, -3, -128, -83, 8, -26, 126, -104, 71, 32, -114, -43, 112, 120, -79, -2, -64, 10, -78, -125, 55, 89, -71, -83, 91, -96, 22, 7, 123, -13, 52, 25, 112, 45, 58, 55, 27, 62, -23, -39, 62, -49, 103, 12, 52, 65, 38, 23, -57, -66, 42, 3, 37, 23, -128, -13, 17, -12, -99, -3, 126, -25, -20, 91, -14, -16, 100, 37, -112, 25, -79, 75, 27, -128, -101, 31, 80, -7, -96, 15, 25, 94, 127, -28, -22, -81, -27, -38, -115, -123, 34, 11, 86, -111, -79, -99, -56, 8, 25, 14, 45, -6, 25, -34, 10, 12, 12, -73, -34, 54, 39, 76, -57, 0, 2, -17, -127, 64, 77, -38, -39, -101, 125, 51, 75, 78, 112, -128, 10, -112, 96, -10, -98, 100, -37, 30, 9, 0, -4, -26, -7, -36, 83, 16, 17, 2, 78, 127, 21, 75, 27, 50, 97, -73, 49, 63, -1, 62, -92, -36, -65, -23, -16, -41, 67, -21, -69, 26, -69, 20, -76, -94, 62, 63, -30, 34, 127, -20, 127, 62, 28, -25, 25, -16, 4, -53, 56, 13, 5, 20, -58, 104, 42, -3, 29, -51, -52, 25, -128, 113, 14, -47, 58, 52, 37, -98, 46, 6, 88, -96, -70, 68, -96, 33, -114, 16, 89, -9, 20, -9, -128, 18, 15, -12, 0, 21, -55, -19, -28, -27, 11, -47, 14, -31, 93, 57, 97, 126, -12, 86, -65, 61, 46, -50, -19, 27, 73, 43, 100, 41, 53, 14, -47, -127, -19, -116, 5, -93, 18, 127, -58, -121, 17, 19, -71, -89, 33, 52, 8, -128, -47, 25, 14, 91, 28, -13, 7, 51, 90, -73, 36, -19, 118, -85, 32, 127, -59
VETOR_ATIVACAO_0: .word 64,32,53,23
NUMERO_CAMADAS: .word 5
.text
.globl _start
# -- _start --
# a0 = código de saída do programa
_start:
    jal main
    li a7, 93
    ecall

# -- main --
main:
    addi sp, sp, -16 # Aloca espaço na pilha
    sw ra, 12(sp)    # Salva o endereço de retorno

    # ----------- Execução da rede neural --------------

    # Todas as variáveis globais devem estar preenchidas exceto VETOR_ATIVACAO_1
    jal irisnet

    # ----------- Impressão do resultado -------------

    # Acha o índice do maior valor no vetor de ativação final
    jal max

    # Escreve o resultado na saída padrão
    jal escrever_resultado

    # -------------- Finalização -----------------

    lw ra, 12(sp)   # Restaura o endereço de retorno
    addi sp, sp, 16 # Desaloca espaço na pilha
    li a0, 0        # retorna 0
    ret
# ------------------------------------------------------------------------------
# Função: irisnet
# Descrição: Função principal da rede neural, chama a função de multiplicação de matriz por vetor para cada camada.
# Argumentos: Nenhum. Usa as variáveis globais.
# Retorno:
#   a0: Ponteiro para o vetor de ativação da última camada.
# ------------------------------------------------------------------------------
irisnet:
    addi sp, sp, -32  # Aloca espaço na pilha
    sw ra, 28(sp)     # Salva o endereço de retorno
    sw s0, 24(sp)     # s0: contador de matrizes a processar (N-1)
    sw s1, 20(sp)     # s1: ponteiro para o array TAM_CAMADAS
    sw s2, 16(sp)     # s2: ponteiro para o array PESOS_MATRIZ
    sw s3, 12(sp)     # s3: ponteiro para o vetor de ativação de ENTRADA da camada
    sw s4, 8(sp)      # s4: ponteiro para o vetor de ativação de SAÍDA da camada
    sw s5, 4(sp)      # s5: salva temporariamente s3 ou s4 para a troca

    # --- Inicialização ---
    la t0, NUMERO_CAMADAS
    lw s0, 0(t0)
    addi s0, s0, -1         # s0 = Número de camadas - 1 (total de matrizes de peso)
    
    la s1, TAM_CAMADAS      # Ponteiro para os tamanhos das camadas
    la s2, PESOS_MATRIZ     # Ponteiro para o início dos pesos
    la s3, VETOR_ATIVACAO_0 # O primeiro vetor de entrada é o VETOR_ATIVACAO_0
    la s4, VETOR_ATIVACAO_1 # O primeiro vetor de saída será o VETOR_ATIVACAO_1

irisnet_loop:
    beqz s0, irisnet_end    # Se o contador de matrizes chegou a zero, termina.

    # --- Prepara os argumentos para a função de multiplicação ---
    lw a0, 0(s1)          # a0 = N_in (tamanho da camada de entrada)
    lw a1, 4(s1)          # a1 = N_out (tamanho da camada de saída)
    mv a2, s2             # a2 = ponteiro para a matriz de pesos atual
    mv a3, s3             # a3 = ponteiro para o vetor de ativação de entrada
    mv a4, s4             # a4 = ponteiro para o vetor de ativação de saída

    jal mult_matriz_vetor_relu # Executa a multiplicação e ReLU para a camada

    # --- Atualização para a próxima iteração ---
    
    # Avança o ponteiro de pesos para a próxima matriz
    lw t0, 0(s1)          # N_in
    lw t1, 4(s1)          # N_out
    mul t2, t0, t1        # Total de pesos na matriz atual = N_in * N_out
    slli t2, t2, 2        # Multiplica por 4 (bytes por inteiro) para obter o tamanho em bytes
    add s2, s2, t2        # s2 aponta para a próxima matriz

    # Avança o ponteiro dos tamanhos de camada
    addi s1, s1, 4

    # Troca os buffers de ativação para a próxima camada
    # VETOR_ATIVACAO_0 e VETOR_ATIVACAO_1 ficam alternando
    # s3 aponta para o vetor de ativação de entrada da próxima camada
    mv s5, s3
    mv s3, s4
    mv s4, s5

    addi s0, s0, -1       # Decrementa o contador de matrizes
    j irisnet_loop

irisnet_end:
    mv a0, s3             # O resultado final está no último buffer apontado por s3

    lw ra, 28(sp)         # Restaura registradores
    lw s0, 24(sp)
    lw s1, 20(sp)
    lw s2, 16(sp)
    lw s3, 12(sp)
    lw s4, 8(sp)
    lw s5, 4(sp)
    addi sp, sp, 32       # Desaloca espaço na pilha
    ret
# ------------------------------------------------------------------------------
# Função: mult_matriz_vetor_relu
# Descrição: Calcula z = W * a e aplica ReLU(z) para uma camada.
# Argumentos:
#   a0: N_in  (número de neurônios na camada de entrada = número de colunas de Wc)
#   a1: N_out (número de neurônios na camada de saída = número de linhas de Wc)
#   a2: Ponteiro para o início da matriz de pesos Wc (N_out x N_in)
#   a3: Ponteiro para o vetor de ativação de entrada 'a'
#   a4: Ponteiro para o vetor de ativação de saída 'z'
# ------------------------------------------------------------------------------
mult_matriz_vetor_relu:
    addi sp, sp, -16
    sw ra, 12(sp)
    sw s6, 8(sp)      # s6: acumulador

    # Loop externo: itera sobre cada neurônio da camada de saída (i)
    li t0, 0 # i = 0
outer_loop:
    beq t0, a1, mult_end # se i == N_out, termina

    # Loop interno: itera sobre cada neurônio da camada de entrada (j)
    li t1, 0 # j = 0
    li s6, 0 # acumulador = 0
    mv t2, a3 # t2 = ponteiro temporário para o vetor 'a', reseta para cada neurônio 'i'
inner_loop:
    beq t1, a0, inner_loop_end # se j == N_in, termina o loop interno

    lw t3, 0(a2)  # t3 = Wij
    lw t4, 0(t2)  # t4 = aj
    
    mul t5, t3, t4
    add s6, s6, t5 # acumulador += Wij * aj

    addi a2, a2, 4 # avança ponteiro W
    addi t2, t2, 4 # avança ponteiro 'a' temporário
    addi t1, t1, 1 # j++
    j inner_loop
inner_loop_end:
    # --- Aplica ReLU ---
    bge s6, zero, relu_end # se acumulador >= 0, pula
    li s6, 0          # se for negativo, zera
relu_end:
    srli s6, s6, 8
    sw s6, 0(a4)      # Salva o resultado final em zi
    addi a4, a4, 4    # avança ponteiro z
    addi t0, t0, 1    # i++
    j outer_loop

mult_end:
    lw ra, 12(sp)      # Restaura registradores
    lw s6, 8(sp)
    addi sp, sp, 16    # Desaloca espaço na pilha
    ret
# ------------------------------------------------------------------------------
# Função: max (Argmax)
# Descrição: Encontra o índice do maior valor no vetor final.
# Argumentos:
#   a0: Ponteiro para o vetor de ativação final.
# Retorno/Efeitos:
#   - Armazena o índice do maior valor no buffer 'OUTPUT'.
# ------------------------------------------------------------------------------
max:
    addi sp, sp, -16
    sw ra, 12(sp)
    sw s1, 8(sp)      # s1: tamanho do vetor (será 3)
    sw s2, 4(sp)      # s2: valor máximo encontrado

    li s1, 3          # O tamanho do vetor final é sempre 3 para este problema
    
    # Inicializa o valor máximo com o primeiro elemento
    lw s2, 0(a0)
    li t0, 0          # t0 = índice atual
    li t1, 0          # t1 = índice do máximo

max_loop:
    beq t0, s1, max_end # Se já verificou todos os elementos, termina
    
    lw t2, 0(a0)      # Carrega o valor atual vi
    ble t2, s2, max_continue # Se vi <= max_val, continua

    # Novo máximo encontrado
    mv s2, t2         # Atualiza o valor máximo
    mv t1, t0         # Atualiza o índice do máximo

max_continue:
    addi a0, a0, 4    # Avança o ponteiro do vetor
    addi t0, t0, 1    # i++
    j max_loop

max_end:
    la t2, OUTPUT
    sw t1, 0(t2)      # Salva o índice final na variável global OUTPUT

    lw ra, 12(sp)
    lw s1, 8(sp)
    lw s2, 4(sp)
    addi sp, sp, 16
    ret
# ------------------------------------------------------------------------------
# Função: escrever_resultado
# Descrição: Imprime o índice final na saída padrão.
# Argumentos: Nenhum. Lê do buffer 'OUTPUT'.
# Retorno/Efeitos: Imprime um número (0, 1 ou 2) e uma nova linha.
# ------------------------------------------------------------------------------
escrever_resultado:
    # Carrega o resultado (0, 1 ou 2) de OUTPUT
    la t0, OUTPUT
    lw a0, 0(t0)

    # Converte o inteiro para seu caractere ASCII correspondente
    addi a0, a0, 48   # Converte para ASCII
    
    # Armazena o caractere no buffer de ASCII para impressão
    la t1, OUTPUT_BUFFER
    sb a0, 0(t1)
    
    # Adiciona uma quebra de linha
    li t2, 10           # '\n'
    sb t2, 1(t1)

    # Prepara a syscall 'write'
    li a7, 64               # Syscall para write
    li a0, 1                # 1 = stdout (saída padrão)
    la a1, OUTPUT_BUFFER    # Ponteiro para a string a ser impressa
    li a2, 2                # Tamanho da string (ex: '1' e '\n')
    ecall
    ret
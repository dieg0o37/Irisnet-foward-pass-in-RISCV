.bss
INPUT_BUFFER: .skip 16384        # Buffer para armazenar a string de entrada inteira
VETOR_ATIVACAO_1: .skip 400     # Buffer para armazenar o vetor de ativação c
VETOR_FINAL: .skip 12           # Buffer para armazenar o vetor final (3 tipos de plantas)
OUTPUT: .skip 4                 # Buffer para armazenar o index da planta escolhida (0, 1 ou 2)
OUTPUT_BUFFER: .skip 2          # Buffer para armazenar o resultado final (0, 1 ou 2) + '\n'
.data
TAM_CAMADAS: .word 4, 8, 15, 3
PESOS_MATRIZ: .word -72, 6, 127, 117, 89, 115, -128, -79, -83, -56, 127, -54, -48, -128, 104, 98, 78, 57, -30, -128, -41, -59, 47, 127, -128, -29, 36, -45, -128, 0, -61, 18, 17, -59, 39, 54, -67, 127, -37, -39, 0, -110, 122, -87, 44, -128, 79, -124, 57, -8, -82, 61, 127, -119, -16, -36, -58, -106, 91, -61, 19, -54, -35, -128, -36, -88, -128, -83, -5, -57, -88, 116, -99, 0, 118, -101, -128, -62, 30, 119, 91, -40, -123, -5, 127, 122, -48, 77, 82, 2, 127, -101, -108, -14, 4, -32, -6, 53, 65, -64, -76, 127, -84, 100, -57, -101, -97, -56, 72, 5, 127, 60, -30, 127, 56, -93, 14, -84, 33, 42, 116, 1, 29, 127, 11, -16, 113, 80, -128, 120, 69, -6, -101, 56, -41, -76, 120, -41, -79, -127, -102, 43, -89, 30, -29, -53, 127, -57, 45, -8, -105, 44, 7, 8, -107, 0, -45, -24, -127, -17, -19, 12, 82, -98, 63, -39, 14, -79, 1, 48, 58, -51, -42, 78, -62, -71, 47, 127, 11, -110, 80, 15, 50, -51, 48, -30, -65, 84, 124, 64, 57, -10, -128, 49, 50, 29, -49
VETOR_ATIVACAO_0: .word 59, 30, 51, 18
NUMERO_CAMADAS: .word 4
.text
.globl _start
# -- _start --
# a0 = código de saída do programa
_start:
    jal main
    li a7, 93
    ecall

# -- inicialização --
ler_input:
    li a0, 0
    li a7, 63
    la a1, INPUT_BUFFER
    li a2, 8192
    ecall
    ret

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
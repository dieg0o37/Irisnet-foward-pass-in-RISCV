.bss
INPUT_BUFFER: .skip 8192        # Buffer para armazenar a string de entrada inteira
TAM_CAMADAS: .skip 20           # Suporta até 5 camadas (5 * 4 bytes/int)
PESOS_MATRIZ: .skip 8192        # Buffer para armazenar os pesos (td em int)
VETOR_ATIVACAO_0: .skip 400     # Buffer para armazenar o vetor de ativação c - 1
VETOR_ATIVACAO_1: .skip 400     # Buffer para armazenar o vetor de ativação c
NUMERO_CAMADAS: .skip 4
OUTPUT: .skip 4                 # Buffer para armazenar o index da planta escolhida (0, 1 ou 2)
OUTPUT_BUFFER: .skip 2          # Buffer para armazenar o resultado final (0, 1 ou 2) + '\n'

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

    # Lê o imput 
    jal ler_input

    # ---------- Processamento da entrada -----------

    # Parsing da primeira linha
    la a0, INPUT_BUFFER
    jal parse_arquitetura
    
    # Parsing da segunda linha
    jal parse_pesos

    # Parsing da terceira linha
    jal parse_vetor_inicial

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
# Função: ler_prox_int
# Descrição: Lê um número inteiro (positivo ou negativo) de uma string.
# Argumentos:
#   a0: Ponteiro para a string (será atualizado).
# Retorno:
#   a0: Ponteiro para depois do número lido.
#   a1: O número inteiro lido.
# ------------------------------------------------------------------------------
ler_prox_int:
    li t0, 0          # Inicializa o número lido como 0
    li t1, 1          # Inicializa o sinal como positivo (-1 = negativo)
    li t2, 10         # Inicializa a base decimal

pular_caracteres:
    lb t3, 0(a0)
    li t4, 45                       # Código ASCII para '-'
    beq t3, t4, sinal_negativo      # Se for '-', trata como negativo
    li t4, 48                       # Código ASCII para '0'
    blt t3, t4, prox_char   
    li t4, 57                       # Código ASCII para '9'
    bgt t3, t4, prox_char           # Se não for entre '0' e '9', pula
    j parse_loop

prox_char:
    addi a0, a0, 1          # Avança o ponteiro
    j pular_caracteres

sinal_negativo:
    li t1, -1               # Define o sinal como negativo
    addi a0, a0, 1          # Avança o ponteiro

parse_loop:
    lb t3, 0(a0)            # Lê o byte atual da string

    li t4, 48               # Código ASCII para '0'
    blt t3, t4, parse_end   # Se for menor que '0', termina o parsing
    li t4, 57               # Código ASCII para '9'
    bgt t3, t4, parse_end   # Se for maior que '9', termina o parsing

    addi t3, t3, -48        # Converte de ASCII para inteiro

    mul t0, t0, t2          # Multiplica o número atual por 10
    add t0, t0, t3          # Adiciona o dígito lido

    addi a0, a0, 1          # Avança o ponteiro
    j parse_loop            # Continua lendo o próximo dígito

parse_end:
    mul t0, t0, t1
    mv a1, t0               # Armazena o número lido em a1
    ret
# ------------------------------------------------------------------------------
# Função: parse_architecture
# Descrição: Analisa a primeira linha da entrada para extrair os tamanhos das camadas.
# Argumentos:
#   a0: Ponteiro para a string de entrada.
# Retorno/Efeitos:
#   Atualiza o vetor TAM_CAMADAS com os tamanhos das camadas.
#   Atualiza NUMERO_CAMADAS com o número de camadas.
#   a0: Ponteiro para a próxima linha da entrada.
# ------------------------------------------------------------------------------
parse_arquitetura:
    addi sp, sp, -16  # Aloca espaço na pilha
    sw ra, 12(sp)     # Salva o endereço de retorno
    sw s0, 8(sp)      # s0: contador de camadas
    sw s1, 4(sp)      # s1: ponteiro para o Array de camadas

    la s1, TAM_CAMADAS  # Ponteiro para o vetor de tamanhos das camadas
    li s0, 0            # Inicializa o contador de camadas

parse_arq_loop:
    jal ler_prox_int  # Lê o próximo inteiro
    sw a1, 0(s1)      # Armazena o tamanho da camada atual
    addi s1, s1, 4    # Avança para o próximo espaço no vetor
    addi s0, s0, 1    # Incrementa o contador de camadas

    # Verifica se tem mais números na linha
    lb t0, 0(a0)                # Lê o próximo caractere
    li t1, 44                   # Código ASCII para ','
    beq t0, t1, parse_arq_loop  # Se for vírgula, continua lendo

prox_linha_arq:
    li t1, 10                   # Código ASCII para '\n'
    lb t0, 0(a0)                # Lê o próximo caractere
    beq t0, t1, fim_parse_arq   # Se for nova linha, termina o parsing
    addi a0, a0, 1              # Avança o ponteiro
    j prox_linha_arq

fim_parse_arq:
    addi a0, a0, 1          # Avança o ponteiro para pular o '\n'
    la t0, NUMERO_CAMADAS   # Ponteiro para o número de camadas
    sw s0, 0(t0)            # Armazena o número de camadas

    lw ra, 12(sp)       # Restaura o endereço de retorno
    lw s0, 8(sp)        # Restaura o contador de camadas
    lw s1, 4(sp)        # Restaura o ponteiro para o Array de camadas
    addi sp, sp, 16     # Desaloca espaço na pilha
    ret                 
# ------------------------------------------------------------------------------
# Função: parse_pesos
# Descrição: Analisa a segunda linha e extrai todos os pesos.
# Argumentos:
#   - a0: Ponteiro para o início da linha de pesos.  
# Retorno/Efeitos:
#   - Atualiza o buffer PESOS_MATRIZ com os pesos lidos.
#   - a0: Ponteiro para a próxima linha da entrada.
# ------------------------------------------------------------------------------
parse_pesos:
    addi sp, sp, -32   # Aloca espaço na pilha
    sw ra, 28(sp)      # salva o endereço de retorno
    sw s0, 24(sp)      # s0: numero de camadas
    sw s1, 20(sp)      # s1: ponteiro para o Array de pesos PESOS_MATRIZ
    sw s3, 16(sp)      # s3: ponteiro para o vetor de tamanhos das camadas
    sw s5, 12(sp)      # s5: número de pesos para cada camada

    la s0, NUMERO_CAMADAS   # Ponteiro para o número de camadas
    lw s0, 0(s0)            # Carrega o número de camadas
    la s1, PESOS_MATRIZ     # Ponteiro para o buffer de pesos
    addi s0, s0, -1         # Inicializa o contador de matrizes de pesos (de há N camadas ent há N-1 matrizes)

    la s3, TAM_CAMADAS      # Ponteiro para o vetor de tamanhos das camadas
    addi a0, a0, 7          # Pula a parte inicial da linha de pesos "{"li":[[" 
parse_pesos_camada_loop:
    lw t1, 0(s3)      # Carrega o tamanho da camada atual
    lw t2, 4(s3)      # Carrega o tamanho da próxima camada
    mul s5, t1, t2    # Calcula o número de pesos para esta camada

parse_camada_loop:
    beqz s5, prox_camada      # Se não há mais pesos, vai para a próxima camada

    jal ler_prox_int          # Lê o próximo inteiro

    sw a1, 0(s1)              # Armazena o peso lido
    addi s1, s1, 4            # Avança para o próximo espaço no buffer de pesos
    addi s5, s5, -1           # Decrementa o contador de pesos restantes
    j parse_camada_loop       # Continua lendo pesos para a camada atual

prox_camada:
    addi s0, s0, -1                 # decrementa o contador de matrizes
    beqz s0, prox_linha_pesos       # Se já leu todas as matrizes, termina

    addi a0, a0, 9                  # Avança o ponteiro para pular a parte final da camada "]],"li":[["
    addi s3, s3, 4                  # Avança para o próximo par de tamanhos de camada
    j parse_pesos_camada_loop       # Continua lendo pesos para a próxima camada

prox_linha_pesos:
    li t1, 10                     # Código ASCII para '\n'
    lb t0, 0(a0)                  # Lê o próximo caractere
    beq t0, t1, fim_parse_pesos   # Se for nova linha, termina o parsing
    addi a0, a0, 1                # Avança o ponteiro
    j prox_linha_pesos            # Continua lendo até encontrar nova linha

fim_parse_pesos:
    addi a0, a0, 1              # Avança o ponteiro para pular o '\n'

    lw ra, 28(sp)       # Restaura os registradores
    lw s0, 24(sp)       
    lw s1, 20(sp)       
    lw s3, 16(sp)       
    lw s5, 12(sp)         
    addi sp, sp, 32     # Desaloca espaço na pilha
    ret            
# ------------------------------------------------------------------------------
# Função: parse_vetor_inicial
# Descrição: Analisa a última linha para obter o vetor de entrada da rede.
# Argumentos:
#   - a0: Ponteiro para o início da linha de entrada.
# Retorno/Efeitos:
#   - Atualiza o buffer VETOR_ATIVACAO_0 com os valores lidos.
# ------------------------------------------------------------------------------
parse_vetor_inicial:
    addi sp, sp, -16  # Aloca espaço na pilha
    sw ra, 12(sp)     # Salva o endereço de retorno
    sw s0, 8(sp)      # s0: número de valores na camada de entrada
    sw s1, 4(sp)      # s1: ponteiro para o vetor de ativação inicial
    sw s2, 0(sp)      # s2: o contador de entradas

    la s0, TAM_CAMADAS      # Ponteiro para o número de valores na camadas
    lw s0, 0(s0)            # Carrega o número de valores na camada de entrada
    la s1, VETOR_ATIVACAO_0 # Ponteiro para o buffer de ativação

    li s2, 0                # Inicializa o contador de entradas

parse_vetor_loop:
    beq s2, s0, fim_parse_vetor  # Se já leu todos os valores, termina
    jal ler_prox_int          # Lê o próximo inteiro
    slli a1, a1, 24
    srai a1, a1, 24
    sw a1, 0(s1)              # Armazena o valor lido no vetor de ativação
    addi s1, s1, 4            # Avança para o próximo espaço no vetor de ativação
    addi s2, s2, 1            # Incrementa o contador de entradas
    j parse_vetor_loop        # Continua lendo valores

fim_parse_vetor:
    lw ra, 12(sp)       # Restaura registradores
    lw s0, 8(sp)        
    lw s1, 4(sp)        
    lw s2, 0(sp)        
    addi sp, sp, 16     # Desaloca espaço na pilha
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
    sw s6, 0(sp)      # s6: igual a 1 para aplicar ReLU, 0 para não aplicar (última multiplicação não aplica ReLU)

    li s6, 1          # Inicializa s6 para aplicar ReLU nas camadas intermediárias

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
    mv a5, s6             # a5 = 1 para aplicar ReLU (exceto na última camada)

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

    li s5, 2
    bne s0, s5, nn_eh_ultima_camada
    li s6, 0          # Reseta s6 para a próxima iteração (não aplica ReLU na última camada)
    nn_eh_ultima_camada:

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
#   a0: N_in  (número de neurônios na camada de entrada = número de colunas de W[c])
#   a1: N_out (número de neurônios na camada de saída = número de linhas de W[c])
#   a2: Ponteiro para o início da matriz de pesos W[c] (N_out x N_in)
#   a3: Ponteiro para o vetor de ativação de entrada 'a'
#   a4: Ponteiro para o vetor de ativação de saída 'z'
#   a5: Se for 1, aplica ReLU(z) (se for 0, não aplica)
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

    lw t3, 0(a2)  # t3 = W[i][j]
    lw t4, 0(t2)  # t4 = a[j]

    mul t5, t3, t4
    add s6, s6, t5 # acumulador += W[i][j] * a[j]
    slli s6, s6, 24
    srai s6, s6, 24

    addi a2, a2, 4 # avança ponteiro W
    addi t2, t2, 4 # avança ponteiro 'a' temporário
    addi t1, t1, 1 # j++
    j inner_loop
inner_loop_end:
    # --- Aplica ReLU ---
    beq a5, zero, relu_end # Se a5 == 0, não aplica ReLU
    bge s6, zero, relu_end # se acumulador >= 0, pula
    li s6, 0          # se for negativo, zera
relu_end:
    sw s6, 0(a4)      # Salva o resultado final em z[i]
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
    
    lw t2, 0(a0)      # Carrega o valor atual v[i]
    ble t2, s2, max_continue # Se v[i] <= max_val, continua

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
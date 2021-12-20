#from timeout_decorator import timeout
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import numpy as np
import time
import os
import psutil 

print("################### Genetic Algorithm Benchmark ###################")
#print("########################### Relatório #############################")
#print('##############################  Menu  #############################')

opcao = 0

while opcao != 5:
  print('##############################  Menu  #############################')
  print('''
  [Cargas de Trabalho:]
  [ 1 ] Baixa
  [ 2 ] Moderada
  [ 3 ] Alta
  [ 4 ] Sair do Benchmark''')
  opcao = int(input('>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Qual é a sua opção: '))

  if opcao == 1:
    tamanho_entrada = int(input('Tamanho das entradas recomendáveis [10 - 20] - Carga de Trabalho BAIXA: '))
    #crescimento_entradas = int(input('Número de incrementosQuantidade de vezes para reperir cada execução'))
    quantidade_execucoes = int(input('Número de incrementos e Quantidade de vezes para reperir cada execução: '))
    tempoTotalDasExecucoes = 0
    tempoTotalDasExecucoes1 = 0
    tempoMedioTotaldasExecucoes = 0

    temposrec = []
    tempospd = []
    print('')

    print("########################### Relatório #############################")
    print("############### Dados Gerais de cada algoritmo ####################")

    # LCS recursivo
    def lcs(X, Y, m, n):
        if m == 0 or n == 0:
            return 0
        elif X[m-1] == Y[n-1]:
            return lcs(X, Y, m-1, n-1) + 1
        else:
            return max(lcs(X, Y, m, n-1), lcs(X, Y, m-1, n))

    # LCS prog dinamica
    def lcs_pd(X, Y, m, n):
      L1 = [[0]*(n + 1) for i in range(m + 1)]
    
      for i in range(m + 1):
          for j in range(n + 1):
              if i == 0 or j == 0 :
                  L1[i][j] = 0
              elif X[i-1] == Y[j-1]:
                  L1[i][j] = L1[i-1][j-1]+1
              else:
                  L1[i][j] = max(L1[i-1][j], L1[i][j-1])
  
      return L1[m][n]

    # tempo de execução do recursivo
    def time_out_re(X,Y,m,n):
      start = time.time()
      lcs(X, Y, m, n)
      end = time.time()
      tempo = end - start
      return tempo

    # tempo de execução do prog dinamica
    def time_out_pd(X,Y,m,n):
      start = time.time()
      lcs_pd(X, Y, m, n)
      end = time.time()
      tempo = end - start
      return tempo

    # função que calcula o tempo total da execução 
    def time_total_de_execucao(X,Y,m,n):
      start_rec = time.time()
      lcs(X, Y, m, n)
      end_rec = time.time()
      tempo_execucao_rec = end_rec - start_rec

      start_pd = time.time()
      lcs_pd(X, Y, m, n)
      end_pd = time.time()
      tempo_execucao_pd = end_pd - start_pd
    
      tempoTotal = (tempo_execucao_rec + tempo_execucao_pd)

      return tempoTotal

    # função de execução da LCS recursivo
    def executor_re(tamanho_entrada):
      conteudo = []
      conteudo2 = []
      conteudo3 = []
      tempos = []
      
      
      temp_m = 0
      medias = 0
      entradas = []
      tamanhos = []
      
      for i in range(quantidade_execucoes):
          V = ["A","T","C","G"]
          W = ""
          for i in range(tamanho_entrada):
              W = W + V[np.random.choice(range(len(V)))]
          meio = len(W)//2
          X = W[:meio]
          Y = W[meio:]
          m = len(X)
          n = len(Y)
          entradas.append((X, Y))
          tamanhos.append(tamanho_entrada)
          temp = time_out_re(X, Y, m, n)
          global tempoTotalDasExecucoes
          tempoTotalDasExecucoes = temp
          tempos.append(temp)
          global temposrec
          temposrec.append(temp)
          temp_m = temp / quantidade_execucoes
          
          medias = (temp_m / quantidade_execucoes)
          tamanho_entrada = tamanho_entrada + 1
    
      print('')
      print("######################### LCS Recursivo #############################")
      print("Maior tamanho da entrada: ", tamanho_entrada)
      print('Quantidade de execuções para cada entrada: ', quantidade_execucoes)
      print('Entradas', entradas)
      print('Tamanhos da Entradas', tamanhos)
      print('Tempos de cada execução', tempos)
      print('Tempo Total da Execução', temp)
      print('Média dos tempos de execução', temp_m)
      print('Uso de CPU em porcentagem: ', psutil.cpu_percent(),'%')
      print('Uso médio de memória RAM em porcentagem: ', psutil.virtual_memory().percent,'%')
      print('')
      print("############################# Gráficos ###############################")
      print('')
      print('Gráfico das Entradas em função dos Tempos')
      x = np.array(tamanhos)
      y = np.array(tempos)

      lx = x
      ly = y

      plt.scatter(x, ly)
      plt.show()

      print('')
      #print('Regressão linear - Gráfico das Entradas em função dos Tempos')
      ##x = np.array(tamanhos)
      #y = np.array(tempos)

      #lx = x
      #ly = np.log2(y)

      #regressão linear, tranformar a curva do gráfico em uma reta a partir de um certo ponto
      #from sklearn.linear_model import LinearRegression
      #model = LinearRegression().fit(lx.reshape(-1, 1), ly)
      #print('slope:', model.coef_)

      #plt.scatter(lx, ly)
      #plt.plot(lx, model.intercept_ + model.coef_ * lx, 'r')
      #plt.show()

      # intervalo de confiança, grau do polinomio
      #import statsmodels.api as sm
      #lx = sm.add_constant(lx)
      #res = sm.OLS(ly, lx).fit()
      #print('slope conf interval:', res.conf_int(0.05)[1])

      print('')
      print('Gráfico das Entradas em função dos Tempos')
      #plt.plot(tamanhos, 'go')  # green bolinha
      #plt.plot(tamanhos, 'k--', color='green')  # linha pontilha

      plt.plot(tempos, 'r^')  # red triangulo
      plt.plot(tempos, 'k--', color='red')  # linha tracejada

      plt.title("Grafico de Desempenho")
      plt.tick_params(axis='x', which='both', bottom=False,
                      top=False, labelbottom=False)
      red_patch = mpatches.Patch(color='red', label='Tempos')
      #green_patch = mpatches.Patch(color='green', label='Tempos')
      plt.legend(handles=[red_patch])

      plt.grid(True)
      plt.xlabel("Quantidade de repeticoes")
      plt.ylabel("Tempos")
      plt.show()

      print('#######################################################################')
    
    # função de execução da LCS recursivo
    def executor_pd(tamanho_entrada):
      conteudo = []
      conteudo2 = []
      conteudo3 = []
      tempos = []

      temp_m = 0
      medias = 0
      entradas = []
      tamanhos = []

    
      #for j in range(crescimento_entradas):
      for i in range(quantidade_execucoes):
          V = ["A","T","C","G"]
          W = ""
          for i in range(tamanho_entrada - 1):
              W = W + V[np.random.choice(range(len(V)))]
          meio = len(W)//2
          X = W[:meio]
          Y = W[meio:]
          m = len(X)
          n = len(Y)
          entradas.append((X, Y))
          tamanhos.append(tamanho_entrada)
          temp = time_out_pd(X, Y, m, n)
          global tempoTotalDasExecucoes1
          tempoTotalDasExecucoes1 = temp
          tempos.append(temp)
          global tempospd
          tempospd.append(temp)
          temp_m = temp / quantidade_execucoes
          medias = (temp_m / quantidade_execucoes)

          
          #global crescimento_entradas
        
          #if():
            #opp = False
            #break
          #else:
          tamanho_entrada = tamanho_entrada + 1

            #print('tamanho entrada', tamanho_entrada)
      
      print('')
      print("####################### LCS Dinâmico ###########################")
      print("Maior Tamanho da entrada: ", tamanho_entrada)
      print('Quantidade de execuções para cada entrada: ', quantidade_execucoes)
      print('Entradas', entradas)
      print('Tamanhos da Entradas', tamanhos)
      print('Tempos de cada execução', tempos)
      print('Tempo Total da Execução', temp)
      print('Média dos tempos de execução', temp_m)
      print('Uso de CPU em porcentagem: ', psutil.cpu_percent(),'%')
      print('Uso médio de memória RAM em porcentagem: ', psutil.virtual_memory().percent,'%')
      print('')
      print("############################# Gráficos ###############################")
      print('')
      print('Gráfico das Entradas em função dos Tempos')
      x = np.array(tamanhos)
      y = np.array(tempos)

      lx = x
      ly = y

      plt.scatter(x, ly)
      plt.show()

      print('')
      #print('Regressão linear - Gráfico das Entradas em função dos Tempos')
      #x = np.array(tamanhos)
      #y = np.array(tempos)

      ##lx = x
      #ly = np.log2(y)

      #regressão linear, tranformar a curva do gráfico em uma reta a partir de um certo ponto
      #from sklearn.linear_model import LinearRegression
      #model = LinearRegression().fit(lx.reshape(-1, 1), ly)
      #print('slope:', model.coef_)

      ##plt.scatter(lx, ly)
      #plt.plot(lx, model.intercept_ + model.coef_ * lx, 'r')
      #plt.show()

      # intervalo de confiança, grau do polinomio
      #import statsmodels.api as sm
      #lx = sm.add_constant(lx)
      #res = sm.OLS(ly, lx).fit()
      #print('slope conf interval:', res.conf_int(0.05)[1])

      print('')
      print('Gráfico das Entradas em função dos Tempos')
      #plt.plot(tamanhos, 'go')  # green bolinha
      #plt.plot(tamanhos, 'k--', color='green')  # linha pontilha

      plt.plot(tempos, 'r^')  # red triangulo
      plt.plot(tempos, 'k--', color='red')  # linha tracejada

      plt.title("Grafico de Desempenho")
      plt.tick_params(axis='x', which='both', bottom=False,
                      top=False, labelbottom=False)
      red_patch = mpatches.Patch(color='red', label='Tempos')
      #green_patch = mpatches.Patch(color='green', label='Tempos')
      plt.legend(handles=[red_patch])

      plt.grid(True)
      plt.xlabel("Quantidade de repeticoes")
      plt.ylabel("Tempos")
      plt.show()

      print('#######################################################################')

    executor_re(tamanho_entrada)
    executor_pd(tamanho_entrada)
    tempoTotalDasExecucoesTotal = tempoTotalDasExecucoes + tempoTotalDasExecucoes1
    tempoMedioTotaldasExecucoes = (tempoTotalDasExecucoesTotal /2)
    print('')
    print('')
    print('')
    print("############### Resultados Gerais ####################")
    print('Tempo total de execução: ', tempoTotalDasExecucoesTotal)
    print('Tempo médio de execução: ', tempoMedioTotaldasExecucoes)
    #print('Variança do tempo de execução: ')
    print('Uso de CPU em porcentagem: ', psutil.cpu_percent(),'%')
    print('Uso médio de memória RAM em porcentagem: ', psutil.virtual_memory().percent,'%')
    print('')
    print("############################# Gráficos ###############################")

    print('')
    print('Gráfico dos Tempos de Execução Recusivo')
    plt.plot(temposrec, 'go')  # green bolinha
    plt.plot(temposrec, 'k--', color='green')  # linha pontilha

    #plt.plot(tempospd, 'r^')  # red triangulo
    #plt.plot(tempospd, 'k--', color='red')  # linha tracejada

    plt.title("Grafico de Desempenho")
    plt.tick_params(axis='x', which='both', bottom=False,
                  top=False, labelbottom=False)
    #red_patch = mpatches.Patch(color='red', label='Tempos REC')
    green_patch = mpatches.Patch(color='green', label='Tempos REC')
    plt.legend(handles=[green_patch])

    plt.grid(True)
    plt.xlabel("Quantidade de repeticoes")
    plt.ylabel("Tempos")
    plt.show()

    
    print('')
    print('Gráfico dos Tempos de Execução Dinâmico')
    #plt.plot(temposrec, 'go')  # green bolinha
    #plt.plot(temposrec, 'k--', color='green')  # linha pontilha

    plt.plot(tempospd, 'r^')  # red triangulo
    plt.plot(tempospd, 'k--', color='red')  # linha tracejada

    plt.title("Grafico de Desempenho")
    plt.tick_params(axis='x', which='both', bottom=False,
                  top=False, labelbottom=False)
    red_patch = mpatches.Patch(color='red', label='Tempos PD')
    #green_patch = mpatches.Patch(color='green', label='Tempos REC')
    plt.legend(handles=[red_patch])

    plt.grid(True)
    plt.xlabel("Quantidade de repeticoes")
    plt.ylabel("Tempos")
    plt.show()

    print('')
    
    #Opções depois do relatório gerado
    print('Opções: ')
    print('''
    [1] - Continuar
    [2] - Sair''')
    
    op = 0
    op = int(input('>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deseja continuar, para mais execuções? '))
    
    if op == 1:
      print('')
      print('Continuando...')
      print('')

    elif op == 2:
      print('')
      print('Saindo...')
      print('')
      break
    
    else:
      print('Opção inválida!!!')
    
  elif opcao == 2:
    tamanho_entrada = int(input('Tamanho das entradas recomendáveis [20 - 30] - Carga de Trabalho BAIXA: '))
    #crescimento_entradas = int(input('Número de incrementosQuantidade de vezes para reperir cada execução'))
    quantidade_execucoes = int(input('Número de incrementos e Quantidade de vezes para reperir cada execução: '))
    tempoTotalDasExecucoes = 0
    tempoTotalDasExecucoes1 = 0
    tempoMedioTotaldasExecucoes = 0

    temposrec = []
    tempospd = []
    print('')

    print("########################### Relatório #############################")
    print("############### Dados Gerais de cada algoritmo ####################")

    # LCS recursivo
    def lcs(X, Y, m, n):
        if m == 0 or n == 0:
            return 0
        elif X[m-1] == Y[n-1]:
            return lcs(X, Y, m-1, n-1) + 1
        else:
            return max(lcs(X, Y, m, n-1), lcs(X, Y, m-1, n))

    # LCS prog dinamica
    def lcs_pd(X, Y, m, n):
      L1 = [[0]*(n + 1) for i in range(m + 1)]
    
      for i in range(m + 1):
          for j in range(n + 1):
              if i == 0 or j == 0 :
                  L1[i][j] = 0
              elif X[i-1] == Y[j-1]:
                  L1[i][j] = L1[i-1][j-1]+1
              else:
                  L1[i][j] = max(L1[i-1][j], L1[i][j-1])
  
      return L1[m][n]

    # tempo de execução do recursivo
    def time_out_re(X,Y,m,n):
      start = time.time()
      lcs(X, Y, m, n)
      end = time.time()
      tempo = end - start
      return tempo

    # tempo de execução do prog dinamica
    def time_out_pd(X,Y,m,n):
      start = time.time()
      lcs_pd(X, Y, m, n)
      end = time.time()
      tempo = end - start
      return tempo

    # função que calcula o tempo total da execução 
    def time_total_de_execucao(X,Y,m,n):
      start_rec = time.time()
      lcs(X, Y, m, n)
      end_rec = time.time()
      tempo_execucao_rec = end_rec - start_rec

      start_pd = time.time()
      lcs_pd(X, Y, m, n)
      end_pd = time.time()
      tempo_execucao_pd = end_pd - start_pd
    
      tempoTotal = (tempo_execucao_rec + tempo_execucao_pd)

      return tempoTotal

    # função de execução da LCS recursivo
    def executor_re(tamanho_entrada):
      conteudo = []
      conteudo2 = []
      conteudo3 = []
      tempos = []
      
      
      temp_m = 0
      medias = 0
      entradas = []
      tamanhos = []
      
      for i in range(quantidade_execucoes):
          V = ["A","T","C","G"]
          W = ""
          for i in range(tamanho_entrada):
              W = W + V[np.random.choice(range(len(V)))]
          meio = len(W)//2
          X = W[:meio]
          Y = W[meio:]
          m = len(X)
          n = len(Y)
          entradas.append((X, Y))
          tamanhos.append(tamanho_entrada)
          temp = time_out_re(X, Y, m, n)
          global tempoTotalDasExecucoes
          tempoTotalDasExecucoes = temp
          tempos.append(temp)
          global temposrec
          temposrec.append(temp)
          temp_m = temp / quantidade_execucoes
          
          medias = (temp_m / quantidade_execucoes)
          tamanho_entrada = tamanho_entrada + 1
    
      print('')
      print("######################### LCS Recursivo #############################")
      print("Maior tamanho da entrada: ", tamanho_entrada)
      print('Quantidade de execuções para cada entrada: ', quantidade_execucoes)
      print('Entradas', entradas)
      print('Tamanhos da Entradas', tamanhos)
      print('Tempos de cada execução', tempos)
      print('Tempo Total da Execução', temp)
      print('Média dos tempos de execução', temp_m)
      print('Uso de CPU em porcentagem: ', psutil.cpu_percent(),'%')
      print('Uso médio de memória RAM em porcentagem: ', psutil.virtual_memory().percent,'%')
      print('')
      print("############################# Gráficos ###############################")
      print('')
      print('Gráfico das Entradas em função dos Tempos')
      x = np.array(tamanhos)
      y = np.array(tempos)

      lx = x
      ly = y

      plt.scatter(x, ly)
      plt.show()

      print('')
      print('Regressão linear - Gráfico das Entradas em função dos Tempos')
      x = np.array(tamanhos)
      y = np.array(tempos)

      lx = x
      ly = np.log2(y)

      #regressão linear, tranformar a curva do gráfico em uma reta a partir de um certo ponto
      from sklearn.linear_model import LinearRegression
      model = LinearRegression().fit(lx.reshape(-1, 1), ly)
      print('slope:', model.coef_)

      plt.scatter(lx, ly)
      plt.plot(lx, model.intercept_ + model.coef_ * lx, 'r')
      plt.show()

      # intervalo de confiança, grau do polinomio
      import statsmodels.api as sm
      lx = sm.add_constant(lx)
      res = sm.OLS(ly, lx).fit()
      print('slope conf interval:', res.conf_int(0.05)[1])

      print('')
      print('Gráfico das Entradas em função dos Tempos')
      #plt.plot(tamanhos, 'go')  # green bolinha
      #plt.plot(tamanhos, 'k--', color='green')  # linha pontilha

      plt.plot(tempos, 'r^')  # red triangulo
      plt.plot(tempos, 'k--', color='red')  # linha tracejada

      plt.title("Grafico de Desempenho")
      plt.tick_params(axis='x', which='both', bottom=False,
                      top=False, labelbottom=False)
      red_patch = mpatches.Patch(color='red', label='Tempos')
      #green_patch = mpatches.Patch(color='green', label='Tempos')
      plt.legend(handles=[red_patch])

      plt.grid(True)
      plt.xlabel("Quantidade de repeticoes")
      plt.ylabel("Tempos")
      plt.show()

      print('#######################################################################')
    
    # função de execução da LCS recursivo
    def executor_pd(tamanho_entrada):
      conteudo = []
      conteudo2 = []
      conteudo3 = []
      tempos = []

      temp_m = 0
      medias = 0
      entradas = []
      tamanhos = []

    
      #for j in range(crescimento_entradas):
      for i in range(quantidade_execucoes):
          V = ["A","T","C","G"]
          W = ""
          for i in range(tamanho_entrada - 1):
              W = W + V[np.random.choice(range(len(V)))]
          meio = len(W)//2
          X = W[:meio]
          Y = W[meio:]
          m = len(X)
          n = len(Y)
          entradas.append((X, Y))
          tamanhos.append(tamanho_entrada)
          temp = time_out_pd(X, Y, m, n)
          global tempoTotalDasExecucoes1
          tempoTotalDasExecucoes1 = temp
          tempos.append(temp)
          global tempospd
          tempospd.append(temp)
          temp_m = temp / quantidade_execucoes
          medias = (temp_m / quantidade_execucoes)

          
          #global crescimento_entradas
        
          #if():
            #opp = False
            #break
          #else:
          tamanho_entrada = tamanho_entrada + 1

            #print('tamanho entrada', tamanho_entrada)
      
      print('')
      print("####################### LCS Dinâmico ###########################")
      print("Maior Tamanho da entrada: ", tamanho_entrada)
      print('Quantidade de execuções para cada entrada: ', quantidade_execucoes)
      print('Entradas', entradas)
      print('Tamanhos da Entradas', tamanhos)
      print('Tempos de cada execução', tempos)
      print('Tempo Total da Execução', temp)
      print('Média dos tempos de execução', temp_m)
      print('Uso de CPU em porcentagem: ', psutil.cpu_percent(),'%')
      print('Uso médio de memória RAM em porcentagem: ', psutil.virtual_memory().percent,'%')
      print('')
      print("############################# Gráficos ###############################")
      print('')
      print('Gráfico das Entradas em função dos Tempos')
      x = np.array(tamanhos)
      y = np.array(tempos)

      lx = x
      ly = y

      plt.scatter(x, ly)
      plt.show()

      print('')
      print('Regressão linear - Gráfico das Entradas em função dos Tempos')
      x = np.array(tamanhos)
      y = np.array(tempos)

      lx = x
      ly = np.log2(y)

      #regressão linear, tranformar a curva do gráfico em uma reta a partir de um certo ponto
      from sklearn.linear_model import LinearRegression
      model = LinearRegression().fit(lx.reshape(-1, 1), ly)
      print('slope:', model.coef_)

      plt.scatter(lx, ly)
      plt.plot(lx, model.intercept_ + model.coef_ * lx, 'r')
      plt.show()

      # intervalo de confiança, grau do polinomio
      import statsmodels.api as sm
      lx = sm.add_constant(lx)
      res = sm.OLS(ly, lx).fit()
      print('slope conf interval:', res.conf_int(0.05)[1])

      print('')
      print('Gráfico das Entradas em função dos Tempos')
      #plt.plot(tamanhos, 'go')  # green bolinha
      #plt.plot(tamanhos, 'k--', color='green')  # linha pontilha

      plt.plot(tempos, 'r^')  # red triangulo
      plt.plot(tempos, 'k--', color='red')  # linha tracejada

      plt.title("Grafico de Desempenho")
      plt.tick_params(axis='x', which='both', bottom=False,
                      top=False, labelbottom=False)
      red_patch = mpatches.Patch(color='red', label='Tempos')
      #green_patch = mpatches.Patch(color='green', label='Tempos')
      plt.legend(handles=[red_patch])

      plt.grid(True)
      plt.xlabel("Quantidade de repeticoes")
      plt.ylabel("Tempos")
      plt.show()

      print('#######################################################################')

    executor_re(tamanho_entrada)
    executor_pd(tamanho_entrada)
    tempoTotalDasExecucoesTotal = tempoTotalDasExecucoes + tempoTotalDasExecucoes1
    tempoMedioTotaldasExecucoes = (tempoTotalDasExecucoesTotal / 5)
    print('')
    print('')
    print('')
    print("############### Resultados Gerais ####################")
    print('Tempo total de execução: ', tempoTotalDasExecucoesTotal)
    print('Tempo médio de execução: ', tempoMedioTotaldasExecucoes)
    #print('Variança do tempo de execução: ')
    print('Uso de CPU em porcentagem: ', psutil.cpu_percent(),'%')
    print('Uso médio de memória RAM em porcentagem: ', psutil.virtual_memory().percent,'%')
    print('')
    print("############################# Gráficos ###############################")

    print('')
    print('Gráfico dos Tempos de Execução Recusivo')
    plt.plot(temposrec, 'go')  # green bolinha
    plt.plot(temposrec, 'k--', color='green')  # linha pontilha

    #plt.plot(tempospd, 'r^')  # red triangulo
    #plt.plot(tempospd, 'k--', color='red')  # linha tracejada

    plt.title("Grafico de Desempenho")
    plt.tick_params(axis='x', which='both', bottom=False,
                  top=False, labelbottom=False)
    #red_patch = mpatches.Patch(color='red', label='Tempos REC')
    green_patch = mpatches.Patch(color='green', label='Tempos REC')
    plt.legend(handles=[green_patch])

    plt.grid(True)
    plt.xlabel("Quantidade de repeticoes")
    plt.ylabel("Tempos")
    plt.show()

    
    print('')
    print('Gráfico dos Tempos de Execução Dinâmico')
    #plt.plot(temposrec, 'go')  # green bolinha
    #plt.plot(temposrec, 'k--', color='green')  # linha pontilha

    plt.plot(tempospd, 'r^')  # red triangulo
    plt.plot(tempospd, 'k--', color='red')  # linha tracejada

    plt.title("Grafico de Desempenho")
    plt.tick_params(axis='x', which='both', bottom=False,
                  top=False, labelbottom=False)
    red_patch = mpatches.Patch(color='red', label='Tempos PD')
    #green_patch = mpatches.Patch(color='green', label='Tempos REC')
    plt.legend(handles=[red_patch])

    plt.grid(True)
    plt.xlabel("Quantidade de repeticoes")
    plt.ylabel("Tempos")
    plt.show()

    print('')
    
    #Opções depois do relatório gerado
    print('Opções: ')
    print('''
    [1] - Continuar
    [2] - Sair''')
    
    op = 0
    op = int(input('>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deseja continuar, para mais execuções? '))
    
    if op == 1:
      print('')
      print('Continuando...')
      print('')

    elif op == 2:
      print('')
      print('Saindo...')
      print('')
      break
    
    else:
      print('Opção inválida!!!')
    
  elif opcao == 3:
    tamanho_entrada = int(input('Tamanho das entradas recomendáveis [30 - 40] - Carga de Trabalho BAIXA: '))
    #crescimento_entradas = int(input('Número de incrementosQuantidade de vezes para reperir cada execução'))
    quantidade_execucoes = int(input('Número de incrementos e Quantidade de vezes para reperir cada execução: '))
    tempoTotalDasExecucoes = 0
    tempoTotalDasExecucoes1 = 0
    tempoMedioTotaldasExecucoes = 0

    temposrec = []
    tempospd = []
    print('')

    print("########################### Relatório #############################")
    print("############### Dados Gerais de cada algoritmo ####################")

    # LCS recursivo
    def lcs(X, Y, m, n):
        if m == 0 or n == 0:
            return 0
        elif X[m-1] == Y[n-1]:
            return lcs(X, Y, m-1, n-1) + 1
        else:
            return max(lcs(X, Y, m, n-1), lcs(X, Y, m-1, n))

    # LCS prog dinamica
    def lcs_pd(X, Y, m, n):
      L1 = [[0]*(n + 1) for i in range(m + 1)]
    
      for i in range(m + 1):
          for j in range(n + 1):
              if i == 0 or j == 0 :
                  L1[i][j] = 0
              elif X[i-1] == Y[j-1]:
                  L1[i][j] = L1[i-1][j-1]+1
              else:
                  L1[i][j] = max(L1[i-1][j], L1[i][j-1])
  
      return L1[m][n]

    # tempo de execução do recursivo
    def time_out_re(X,Y,m,n):
      start = time.time()
      lcs(X, Y, m, n)
      end = time.time()
      tempo = end - start
      return tempo

    # tempo de execução do prog dinamica
    def time_out_pd(X,Y,m,n):
      start = time.time()
      lcs_pd(X, Y, m, n)
      end = time.time()
      tempo = end - start
      return tempo

    # função que calcula o tempo total da execução 
    def time_total_de_execucao(X,Y,m,n):
      start_rec = time.time()
      lcs(X, Y, m, n)
      end_rec = time.time()
      tempo_execucao_rec = end_rec - start_rec

      start_pd = time.time()
      lcs_pd(X, Y, m, n)
      end_pd = time.time()
      tempo_execucao_pd = end_pd - start_pd
    
      tempoTotal = (tempo_execucao_rec + tempo_execucao_pd)

      return tempoTotal

    # função de execução da LCS recursivo
    def executor_re(tamanho_entrada):
      conteudo = []
      conteudo2 = []
      conteudo3 = []
      tempos = []
      
      
      temp_m = 0
      medias = 0
      entradas = []
      tamanhos = []
      
      for i in range(quantidade_execucoes):
          V = ["A","T","C","G"]
          W = ""
          for i in range(tamanho_entrada):
              W = W + V[np.random.choice(range(len(V)))]
          meio = len(W)//2
          X = W[:meio]
          Y = W[meio:]
          m = len(X)
          n = len(Y)
          entradas.append((X, Y))
          tamanhos.append(tamanho_entrada)
          temp = time_out_re(X, Y, m, n)
          global tempoTotalDasExecucoes
          tempoTotalDasExecucoes = temp
          tempos.append(temp)
          global temposrec
          temposrec.append(temp)
          temp_m = temp / quantidade_execucoes
          
          medias = (temp_m / quantidade_execucoes)
          tamanho_entrada = tamanho_entrada + 1
    
      print('')
      print("######################### LCS Recursivo #############################")
      print("Maior tamanho da entrada: ", tamanho_entrada)
      print('Quantidade de execuções para cada entrada: ', quantidade_execucoes)
      print('Entradas', entradas)
      print('Tamanhos da Entradas', tamanhos)
      print('Tempos de cada execução', tempos)
      print('Tempo Total da Execução', temp)
      print('Média dos tempos de execução', temp_m)
      print('Uso de CPU em porcentagem: ', psutil.cpu_percent(),'%')
      print('Uso médio de memória RAM em porcentagem: ', psutil.virtual_memory().percent,'%')
      print('')
      print("############################# Gráficos ###############################")
      print('')
      print('Gráfico das Entradas em função dos Tempos')
      x = np.array(tamanhos)
      y = np.array(tempos)

      lx = x
      ly = y

      plt.scatter(x, ly)
      plt.show()

      print('')
      print('Regressão linear - Gráfico das Entradas em função dos Tempos')
      x = np.array(tamanhos)
      y = np.array(tempos)

      lx = x
      ly = np.log2(y)

      #regressão linear, tranformar a curva do gráfico em uma reta a partir de um certo ponto
      from sklearn.linear_model import LinearRegression
      model = LinearRegression().fit(lx.reshape(-1, 1), ly)
      print('slope:', model.coef_)

      plt.scatter(lx, ly)
      plt.plot(lx, model.intercept_ + model.coef_ * lx, 'r')
      plt.show()

      # intervalo de confiança, grau do polinomio
      import statsmodels.api as sm
      lx = sm.add_constant(lx)
      res = sm.OLS(ly, lx).fit()
      print('slope conf interval:', res.conf_int(0.05)[1])

      print('')
      print('Gráfico das Entradas em função dos Tempos')
      #plt.plot(tamanhos, 'go')  # green bolinha
      #plt.plot(tamanhos, 'k--', color='green')  # linha pontilha

      plt.plot(tempos, 'r^')  # red triangulo
      plt.plot(tempos, 'k--', color='red')  # linha tracejada

      plt.title("Grafico de Desempenho")
      plt.tick_params(axis='x', which='both', bottom=False,
                      top=False, labelbottom=False)
      red_patch = mpatches.Patch(color='red', label='Tempos')
      #green_patch = mpatches.Patch(color='green', label='Tempos')
      plt.legend(handles=[red_patch])

      plt.grid(True)
      plt.xlabel("Quantidade de repeticoes")
      plt.ylabel("Tempos")
      plt.show()

      print('#######################################################################')
    
    # função de execução da LCS recursivo
    def executor_pd(tamanho_entrada):
      conteudo = []
      conteudo2 = []
      conteudo3 = []
      tempos = []

      temp_m = 0
      medias = 0
      entradas = []
      tamanhos = []

    
      #for j in range(crescimento_entradas):
      for i in range(quantidade_execucoes):
          V = ["A","T","C","G"]
          W = ""
          for i in range(tamanho_entrada - 1):
              W = W + V[np.random.choice(range(len(V)))]
          meio = len(W)//2
          X = W[:meio]
          Y = W[meio:]
          m = len(X)
          n = len(Y)
          entradas.append((X, Y))
          tamanhos.append(tamanho_entrada)
          temp = time_out_pd(X, Y, m, n)
          global tempoTotalDasExecucoes1
          tempoTotalDasExecucoes1 = temp
          tempos.append(temp)
          global tempospd
          tempospd.append(temp)
          temp_m = temp / quantidade_execucoes
          medias = (temp_m / quantidade_execucoes)

          
          #global crescimento_entradas
        
          #if():
            #opp = False
            #break
          #else:
          tamanho_entrada = tamanho_entrada + 1

            #print('tamanho entrada', tamanho_entrada)
      
      print('')
      print("####################### LCS Dinâmico ###########################")
      print("Maior Tamanho da entrada: ", tamanho_entrada)
      print('Quantidade de execuções para cada entrada: ', quantidade_execucoes)
      print('Entradas', entradas)
      print('Tamanhos da Entradas', tamanhos)
      print('Tempos de cada execução', tempos)
      print('Tempo Total da Execução', temp)
      print('Média dos tempos de execução', temp_m)
      print('Uso de CPU em porcentagem: ', psutil.cpu_percent(),'%')
      print('Uso médio de memória RAM em porcentagem: ', psutil.virtual_memory().percent,'%')
      print('')
      print("############################# Gráficos ###############################")
      print('')
      print('Gráfico das Entradas em função dos Tempos')
      x = np.array(tamanhos)
      y = np.array(tempos)

      lx = x
      ly = y

      plt.scatter(x, ly)
      plt.show()

      print('')
      print('Regressão linear - Gráfico das Entradas em função dos Tempos')
      x = np.array(tamanhos)
      y = np.array(tempos)

      lx = x
      ly = np.log2(y)

      #regressão linear, tranformar a curva do gráfico em uma reta a partir de um certo ponto
      from sklearn.linear_model import LinearRegression
      model = LinearRegression().fit(lx.reshape(-1, 1), ly)
      print('slope:', model.coef_)

      plt.scatter(lx, ly)
      plt.plot(lx, model.intercept_ + model.coef_ * lx, 'r')
      plt.show()

      # intervalo de confiança, grau do polinomio
      import statsmodels.api as sm
      lx = sm.add_constant(lx)
      res = sm.OLS(ly, lx).fit()
      print('slope conf interval:', res.conf_int(0.05)[1])

      print('')
      print('Gráfico das Entradas em função dos Tempos')
      #plt.plot(tamanhos, 'go')  # green bolinha
      #plt.plot(tamanhos, 'k--', color='green')  # linha pontilha

      plt.plot(tempos, 'r^')  # red triangulo
      plt.plot(tempos, 'k--', color='red')  # linha tracejada

      plt.title("Grafico de Desempenho")
      plt.tick_params(axis='x', which='both', bottom=False,
                      top=False, labelbottom=False)
      red_patch = mpatches.Patch(color='red', label='Tempos')
      #green_patch = mpatches.Patch(color='green', label='Tempos')
      plt.legend(handles=[red_patch])

      plt.grid(True)
      plt.xlabel("Quantidade de repeticoes")
      plt.ylabel("Tempos")
      plt.show()

      print('#######################################################################')

    executor_re(tamanho_entrada)
    executor_pd(tamanho_entrada)
    tempoTotalDasExecucoesTotal = tempoTotalDasExecucoes + tempoTotalDasExecucoes1
    tempoMedioTotaldasExecucoes = (tempoTotalDasExecucoesTotal / 5)
    print('')
    print('')
    print('')
    print("############### Resultados Gerais ####################")
    print('Tempo total de execução: ', tempoTotalDasExecucoesTotal)
    print('Tempo médio de execução: ', tempoMedioTotaldasExecucoes)
    #print('Variança do tempo de execução: ')
    print('Uso de CPU em porcentagem: ', psutil.cpu_percent(),'%')
    print('Uso médio de memória RAM em porcentagem: ', psutil.virtual_memory().percent,'%')
    print('')
    print("############################# Gráficos ###############################")

    print('')
    print('Gráfico dos Tempos de Execução Recusivo')
    plt.plot(temposrec, 'go')  # green bolinha
    plt.plot(temposrec, 'k--', color='green')  # linha pontilha

    #plt.plot(tempospd, 'r^')  # red triangulo
    #plt.plot(tempospd, 'k--', color='red')  # linha tracejada

    plt.title("Grafico de Desempenho")
    plt.tick_params(axis='x', which='both', bottom=False,
                  top=False, labelbottom=False)
    #red_patch = mpatches.Patch(color='red', label='Tempos REC')
    green_patch = mpatches.Patch(color='green', label='Tempos REC')
    plt.legend(handles=[green_patch])

    plt.grid(True)
    plt.xlabel("Quantidade de repeticoes")
    plt.ylabel("Tempos")
    plt.show()

    
    print('')
    print('Gráfico dos Tempos de Execução Dinâmico')
    #plt.plot(temposrec, 'go')  # green bolinha
    #plt.plot(temposrec, 'k--', color='green')  # linha pontilha

    plt.plot(tempospd, 'r^')  # red triangulo
    plt.plot(tempospd, 'k--', color='red')  # linha tracejada

    plt.title("Grafico de Desempenho")
    plt.tick_params(axis='x', which='both', bottom=False,
                  top=False, labelbottom=False)
    red_patch = mpatches.Patch(color='red', label='Tempos PD')
    #green_patch = mpatches.Patch(color='green', label='Tempos REC')
    plt.legend(handles=[red_patch])

    plt.grid(True)
    plt.xlabel("Quantidade de repeticoes")
    plt.ylabel("Tempos")
    plt.show()

    
    print('')
    #Opções depois do relatório gerado
    print('Opções: ')
    print('''
    [1] - Continuar
    [2] - Sair''')
    
    op = 0
    op = int(input('>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deseja continuar, para mais execuções? '))
    
    if op == 1:
      print('')
      print('Continuando...')
      print('')

    elif op == 2:
      print('')
      print('Saindo...')
      print('')
      break
    
    else:
      print('Opção inválida!!!')
    

  elif opcao == 4:
    break
    print('#################### Fim do Benchmark! Volte Sempre!!! ####################')

  else:
    print('Opção inválida!!!')

print('')
print('=-=' * 25)

print('#################### Fim do Benchmark! Volte Sempre!!! ####################')

import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import statistics as st
import matplotlib.pyplot as plt
import math
import os
from sklearn import tree
from sklearn.tree import export_graphviz

def lista_arquivos(diretorio='./', tipo=".txt"):
    caminhos = [os.path.join(nome, diretorio) for nome in os.listdir(diretorio)]
    
    arquivos = []
    for caminho, _, arquivo in os.walk(diretorio):
        for arq in arquivo:
            arquivos.append(caminho+'/'+arq)
    
    lista = [arq for arq in arquivos if arq.lower().endswith(tipo)]
    #lista = ['./dados/'+lista[n] for n in range(0,len(lista))]
    
    return lista

def abrir_arquivos(arq, qtd):
    n = 0
    dados = []
    while n < qtd:
        val = np.genfromtxt(arq[n], delimiter=',')
        dados.append(val)
        n+=1
    return dados

def abrir_bin(arq, qtd):
    n = 0
    dados = []
    while n < qtd:
        val = np.fromfile(arq[n], dtype=">u2")
        dados.append(val)
        n+=1
    return dados

def energia_sinal(sinal, block_size = 64):
    N1 = len(sinal)-1
    n = 0
    j = ecg_full
    energia = []
    while(n < N1):
        i = 0
        val = 0
        while(i < 2*block_size):
            if((n+i) >= N1):
                break

            val += sinal[n+i]**2
            i+=1
        while(j < n):
            energia.append(val/i)
            j+=1

        n+=block_size
    while(j <= N1):
        energia.append(val/i)
        j+=1

    energia = np.array(energia)
    
    return energia

def detect_ativ(sinal, th, block_size = 64):
    
    tr = th*np.mean(sinal)
    ativacao = np.zeros(len(sinal))
    ativ_i = [0]
    j = block_size
    i = 0
    val = 0
    flag_ativ = 0
    
    while(j < (len(sinal)-1)):
        while((i < j) and (i < (len(sinal)-1))):
            if( (j+2*block_size) < (len(sinal)-1) ):
                if((sinal[j] > tr) and (sinal[j+block_size] > tr) and (sinal[j+2*block_size] > tr)):
                    if((sinal[j-block_size] < tr) and (sinal[j-2*block_size] < tr)):
                        val = 1
                        if(flag_ativ == 0):
                            ativ_i.append(i)
                            flag_ativ = 1
                        
            if( (j+2*block_size) < (len(sinal)-1) ):
                if((sinal[j] > tr) and (sinal[j-block_size] > tr) and (sinal[j-2*block_size] > tr)):
                    if((sinal[j+block_size] < tr) and (sinal[j+2*block_size] < tr)):
                        val = 0
                        if(flag_ativ == 1):
                            ativ_i.append(i)
                            flag_ativ = 0
                        
            ativacao[i] = val
            i += 1
        j+=block_size
        
    ativ_i.append(len(sinal)-1)
    return ativacao, ativ_i, tr

def fft_sinal(sinal, fs):
    fft_sinal = np.fft.fft(sinal)
    freq_sinal = np.fft.fftfreq(sinal.size, 1/fs)
    
    return freq_sinal, fft_sinal

def filtro_sinal(sinal, freq_low, freq_high, fs, ordem=5, rp = 0):
    
    nyq = fs/2
    
    if rp == 0:
        bh, ah = signal.bessel(ordem, freq_low/nyq, btype='highpass', analog=False)
        bl, al = signal.bessel(ordem, freq_high/nyq, btype='lowpass', analog=False)
    else:
        bh, ah = signal.cheby1(ordem, rp, freq_low/nyq, btype='highpass', analog=False)
        bl, al = signal.cheby1(ordem, rp, freq_high/nyq, btype='lowpass', analog=False)
    
    sinal_ = signal.filtfilt(bl, al, sinal)
    result = signal.filtfilt(bh, ah, sinal_)
    
    
    return result

def separa_sinal(temp, sinal, pontos):
    
    bloco_x = []
    bloco_y = []
    qtd_blocos = (len(pontos)-1)
    i = 0
    while(i < qtd_blocos):
        bloco_y.append(sinal[pontos[i]:pontos[i+1]])
        bloco_x.append(temp[pontos[i]:pontos[i+1]])
        i+=1
    
    return [bloco_x, bloco_y]

def sinal_media(sinal, block_size = 64):
    
    media = np.zeros(len(sinal))
    j = block_size
    i = 0
    
    while(j < (len(sinal)-1)):
        while((i < j) and (i < (len(sinal)-1))):
            if( (j+2*block_size) < (len(sinal)-1) ):
                media[i] = np.mean(np.abs(sinal[j:j+block_size]))
            i += 1
        j+=block_size
        
    return media

def sinal_rms(sinal, block_size = 64):
    
    rms = np.zeros(len(sinal))
    j = block_size
    i = 0
    
    while(j < (len(sinal)-1)):
        while((i < j) and (i < (len(sinal)-1))):
            if( (j+2*block_size) < (len(sinal)-1) ):
                rms[i] = np.sqrt(np.mean(sinal[j:j+block_size]**2))
            i += 1
        j+=block_size
        
    return rms

def sinal_squaring(sinal):
        
    return sinal**2

def sinal_picos(sinal, th = 150, prominence=0):
    
    media = np.mean(sinal)
    peaks, _ = signal.find_peaks(sinal, height=media, distance=th, prominence=prominence)
    
    return peaks

def sinal_vales(sinal, th = 150, prominence=0):
    
    media = np.mean(-1*sinal)
    peaks, _ = signal.find_peaks(-1*sinal, distance=th, prominence=prominence)
    
    return peaks

def ppg_init(ppg_x, ppg, ecg_x, ecg, filtro = 0):
    
    df_ppg = np.zeros(ppg.size)
    
    pk_ecg = sinal_picos(ecg, prominence=1)
    #vl_ppg = sinal_vales(ppg, prominence=0)
    df_ppg[0] = 0
    df_ppg[1:ppg.size] = np.diff(ppg)/np.diff(ppg_x)
    
    if filtro == 1:
        #fig, ax = plt.subplots(1, 1)
        #plt.plot(ppg_x, df_ppg)
        df_ppg = moving_average(df_ppg, 5, 5, 10);
        #plt.plot(ppg_x, df_ppg)
    
    pk_df_ppg = sinal_picos(df_ppg, prominence=7)
    
### COM PROBLEMA NOS BATIMENTOS ECTOPICOS ###
#    if(pk_df_ppg.size > pk_ecg.size):
#        pk_df_ppg = pk_df_ppg[0:pk_ecg.size]
        
#    if(pk_ecg.size > pk_df_ppg.size):
#        pk_ecg = pk_ecg[0:pk_df_ppg.size]

#    result_y = ppg_x[pk_df_ppg] - ecg_x[pk_ecg]
#    result_x = ppg_x[pk_df_ppg]
#############################################

### PARA REMOVER O BATIMENTO ECTOPICO ###
    #print("PPG: %d \nECG: %d\n" % (pk_df_ppg.size, pk_ecg.size))
    i = 0
    j = 0
    maior = max([pk_ecg.size, pk_df_ppg.size])
    result_y = np.zeros(maior)
    while((j < pk_ecg.size) and (i < pk_df_ppg.size)):
        result_y[i] = ppg_x[pk_df_ppg[i]] - ecg_x[pk_ecg[j]]
        
        if(i>0):
            if(result_y[i]>1.5*result_y[i-1]):
                result_y[i] = 0
                i -= 1
        
        i += 1
        j += 1
        
    if(pk_ecg.size > pk_df_ppg.size):
        result_x = ecg_x[pk_ecg]
    else: 
        result_x = ppg_x[pk_df_ppg]
#########################################

########## DEPURACAO DA FUNCAO ##########
    #plt.plot(ppg_x, df_ppg)
    #plt.plot(ppg_x[pk_df_ppg], df_ppg[pk_df_ppg], '.')
    #plt.plot(ppg_x, ppg)
    #plt.plot(ppg_x[pk_df_ppg], ppg[pk_df_ppg], '.')
    #plt.plot(ecg_x, ecg)
    #plt.plot(ecg_x[pk_ecg], ecg[pk_ecg], '.')
#########################################

    return result_y, result_x

def moving_average(data_set, para_tras, para_frente, const = 100):
    new_data = list()
    delta = para_frente + para_tras
    for j in range(para_tras):
        new_data.append(0)
    for i in range(para_tras ,len(data_set) - para_frente):
        buffer = 0
        for j in range(delta):
            buffer = buffer + (data_set[i - para_tras + j])
        new_data.append(buffer/delta*const)
       
    for j in range(para_frente):
        new_data.append(0)
    return new_data

def sinal_variancia(sinal, block_size = 64):
    
    media = np.zeros(len(sinal))
    j = block_size
    i = 0
    flag_ativ = 0
    
    while(j < (len(sinal)-1)):
        while((i < j) and (i < (len(sinal)-1))):
            if( (j+2*block_size) < (len(sinal)-1) ):
                media[i] = np.var(np.abs(sinal[j:j+block_size]))
            i += 1
        j+=block_size
        
    return media

def sinal_periodo(sinal):
    
    wl = np.zeros(len(sinal))
    i = 0
    
    while(i < (len(sinal)-1)):
        wl[i] = sinal[i+1] - sinal[i]
        
        if(i>0):
            if(wl[i] < 0.8*wl[i-1]):
                wl[i] = wl[i-1]
            if(wl[i] > 1.2*wl[i-1]):
                wl[i] = wl[i-1]
        i += 1
    wl[i] = 0
    
    return wl

def ar_pred(train, test):
    model = AR(train).fit()
    window = model.k_ar
    coef = model.params

    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]

    predictions = list()
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0]
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1]
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
    
    return predictions

def auto_reg(sinal, n):
    
    lst_n = len(sinal)
    x = np.zeros((n, lst_n))
    i = 1
    while(i <= n):
        x[i-1, i:lst_n] = sinal[0:lst_n-i]
        i += 1
    
    vet_a = np.zeros((lst_n, n))
    val_var = np.zeros(lst_n)
    i = 0
    y = []
    while(i < n):
        val_x = x[0:i+1,:]
        matriz_x = np.matmul(val_x, val_x.transpose())
        inv_x = np.linalg.inv(matriz_x)
        
        if( i > 0):
            vet_a[i, 0:i+1] = np.matmul(sinal, np.matmul(val_x.transpose(), inv_x))
        else:
            vet_a[i, 0] = np.matmul(sinal, inv_x * val_x.transpose())
        
        val_var[i] = np.var( sinal - np.matmul(vet_a[i, 0:i+1], x[0:i+1, :]))
        mdl = lst_n*np.log10(val_var[i]) + i*np.log10(lst_n)
        y.append(mdl)
        
        i += 1

    #plt.plot(y)
    
    sinal_criado = np.matmul(vet_a[np.argmin(y), 0:np.argmin(y)+1], x[0:np.argmin(y)+1, :])
    
    return np.argmin(y), vet_a[np.argmin(y), 0:np.argmin(y)], y, val_var[np.argmin(y)], sinal_criado


def arma(pwtt, rr, p):
    
    lst_n = len(pwtt)
    x = np.zeros((lst_n, 2*p))
    x_pwtt = np.zeros((lst_n, p))
    x_rr = np.zeros((lst_n, p))
    
    i = 1
    while(i <= p):
        x[i:lst_n, i-1] = rr[0:lst_n-i]
        x_rr[i:lst_n, i-1] = rr[0:lst_n-i]
        i += 1
    
    i = 0
    while(i < p):
        x[i:lst_n, i+p] = pwtt[0:lst_n-i]
        x_pwtt[i:lst_n, i] = pwtt[0:lst_n-i]
        i += 1
    
    p_i = 0
    
    rr_pwtt = np.concatenate([rr, pwtt])
    ab = []
    while(p_i < p):
        
        ab.append(gera_vetas(rr_pwtt, x, p_i))
        p_i += 1
        
    y = []
    MDL = []
    for p_i in range(p):
        
        y_1 = np.matmul(x_rr[:][0:p_i], ab[p_i][0:p])
        y_2 = np.matmul(x_pwtt[:][0:p_i], ab[p_i][p:2*p])
        
        
        y.append(y_1 + y_2)
        e = rr - y[p_i]
        MDL.append(np.var(e))
        MDL[p_i] = N*np.log(MDL[p_i]) + p*np.log(lst_n)
    
    return MDL


def arma2(pwtt, rr, p):
    mx_pwtt, veta_pwtt = gera_mx_veta(pwtt, p, 0)
    mx_rr, veta_rr = gera_mx_veta(rr, p)
    
    
    
def gera_mx_veta(sinal, P=8, n=1):
    N = len(sinal)
    MDL = []
    y = []
    mx = geraY(sinal, P, n)
    veta = gera_vetas(sinal, mx, P)
    return mx, veta
    
    #for p in range(P):
    #    y.append(np.matmul(mx[p], veta[p]))
    #    e = sinal - y[p]
    #    MDL.append(np.var(e))
    #    MDL[p] = N*np.log(MDL[p]) + p*np.log(N)
    #plt.plot(MDL)
   
    #ordem = np.argmin(MDL)
   
    #print("A ordem de melhor resultado é",ordem)
    #print("Conficientes veta", veta[ordem])
    #%% plot do sinal em comparação com o modelo auto regressivo
    #for i in range(len(MDL)):
        #comparePlot(range(len(sinal)),sinal,range(len(y[i])),y[i])
   
    
    
def geraY(sinal, p = 100, n = 1):
    N = len(sinal)
    y = []
    for P in range(p):
        #print(P)
        y.append(np.zeros((N,P+1)))
        for C in range(P+1):
            for L in range(N-C-n):
                y[P][L+C+n][C] = sinal[L]
        #print(y[P])
    return y

    
    
def gera_vetas(sinal, mx, P):
    veta = []
    
    for p in range(P):
        
        mxt = np.transpose(mx[p])
        mxn = mx[p]
        veta.append(np.matmul(np.matmul(np.linalg.inv(np.matmul(mxt, mxn)),mxt),sinal))
        #print(veta[p])
    return veta
    
    
    
    
    

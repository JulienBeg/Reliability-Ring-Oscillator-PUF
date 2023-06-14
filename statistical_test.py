import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from puf_dataset import *
from scipy.stats import norm,shapiro,probplot
from scipy.signal import periodogram

import tikzplotlib

def kolmogorov_smirnov_test(f,save=False):

    df = load(f)

    Winmax = df["winsize"].max()
    df = df[df["winsize"]==Winmax]

    Delta = df.groupby(["pin_setting","index"]).agg("mean")["delta_freq"].to_numpy()
    Delta = np.reshape(Delta,-1)
    Delta -= np.mean(Delta)
    Delta /= np.std(Delta)

    x = np.linspace(-5,5,10**4)
    Fnorm = norm.cdf(x)
    Fdelta = np.mean( Delta.reshape((len(Delta),1)) < x.reshape(1,10**4),axis=0)

    probplot(Delta,dist='norm',fit=True,plot=plt)
    plt.title("")
    plt.grid()
    if save:
        plt.savefig(f+"_QQplot.pdf",format="pdf",dpi=600)
    plt.show()

    print("For Kolmogorov Smirnov Test : ")
    print(np.max(np.abs(Fnorm-Fdelta)))

    print("Shapiro Test : ")
    print(shapiro(Delta))

    plt.plot(x,Fnorm)
    plt.plot(x,Fdelta)
    plt.grid()
    plt.show()


def kolmogorov_smirnov_test_folder(F,name="dummy_name.tex"):

    Delta = []
    for f in F:
        if "BOARD3" in f:
            df = load(f)

            print("Treating " + f)
            delta = df["delta_freq"].to_numpy().reshape((-1,64))/df["number_measurments"].to_numpy().reshape((-1,64))
            d = np.mean(delta,axis=0)[0:64]
            Delta.append(d) 

    Delta = np.reshape(Delta,-1)
    Delta -= np.mean(Delta)
    print(np.std(Delta))
    Delta /= np.std(Delta)

    x = np.linspace(-5,5,10**4)
    Fnorm = norm.cdf(x)
    Fdelta = np.mean( Delta.reshape((len(Delta),1)) < x.reshape(1,10**4),axis=0)

    probplot(Delta,dist='norm',fit=True,plot=plt)
    plt.title("")
    plt.grid()
    
    ### Takes To Much Memory ###
    #tikzplotlib.clean_figure() 
    #tikzplotlib.save(name)
    #plt.savefig("QQplot_delay_"+name+".pdf",dpi=600,format="pdf")
    plt.savefig(name) 

    plt.show()

    print("For Kolmogorov Smirnov Test : ")
    print(np.max(np.abs(Fnorm-Fdelta)))

    print("Shapiro Test : ")
    print(shapiro(Delta))

    return Delta

def noise(df,save=False,save2=False,name="dummy_name"):

    Noise = []
    for i in range(64):
        noise = df[(df["winsize"]==2**19) * (df["index"]==i) * (df["pin_setting"]==0)]["delta_freq"].to_numpy().astype("float")
        noise -= np.mean(noise)
        noise /= np.std(noise)
        Noise.append(noise)
    Noise = np.array(Noise).reshape(-1)

    probplot(Noise,dist='norm',fit=True,plot=plt)
    plt.title("")
    plt.grid()
    if save:
        plt.savefig("QQplot_noise_"+name,dpi=600,format="pdf")
    plt.show()

    print("With Shapiro Test : ")
    print(shapiro(Noise))

    x = np.linspace(-5,5,10**4)
    Fnorm = norm.cdf(x)
    Fnoise = np.mean( Noise.reshape((len(Noise),1)) < x.reshape(1,10**4),axis=0)
    print("For Kolmogorov Smirnov Test : ")
    print(np.max(np.abs(Fnorm-Fnoise)))

    fs = 400 * 10 ** 6 / 2**19 / 2
    f,P = periodogram(Noise,fs)
    plt.plot(f,P)
    plt.ylabel("Power Spectral Density of Noise")
    plt.xlabel("Frequency in Hz")
    plt.grid()
    if save2:
        plt.savefig("DSPnoise_"+name,dpi=600,format="pdf")
    plt.show()

    return Noise


def time_cb(F):

    Rep = []
    for f in F:
        if ("puf_6" not in f):
            df = load(f)

            print("Treating " + f)
            rep = df["number_measurments"].to_numpy().reshape((-1,64))
            Rep.append(np.mean(rep,axis=0))

    Rep = np.reshape(Rep,-1)

    return Rep

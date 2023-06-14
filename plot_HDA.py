import os.path 
import numpy as np 
import matplotlib.pyplot as plt 
import tikzplotlib

Labels =["WITHOUT_HDA","1PCK","2PCK","4PCK","8PCK","SECDED_ALG","SECDED_ML","SHA256"]
MOD = ["SQRT"] #,"CB"] 
TEMP = [30,40] #,50,60]
BOARDS = 1+np.arange(5)
PUFS = np.arange(16)
PINS = np.arange(8)


"""
Compute CDF of BERs
"""
def cdf_ber(B):
    BER = np.linspace(0,0.1,10**5).reshape((1,10**5))
    return np.mean( np.array(B).reshape((len(B),1)) <= BER, axis=0)

"""
Compute CDF of KERs
"""
def cdf_ker(K):
    KER = np.linspace(0,1,10**4).reshape((1,10**4))
    return np.mean( np.array(K).reshape((len(K),1)) <= KER, axis=0)

def plot_KER(mod="CB",temp=30):
    proba = np.linspace(0,1,10**4)

    for label in Labels:
        cpath = f"../../puf-no-hda/HDA_RESULTS_CORR/{mod}/{temp}C/{label}/"
        ker = np.load(cpath+"KER.npy")
        ker_cdf = cdf_ker(1-ker)
        plt.plot(1-ker_cdf,1-proba,label=label)
        print(f"Average KER {label} : ", np.mean(ker) )
        print(640-len(ker))
    plt.grid()
    plt.xlabel("Survival Distribution Function")
    plt.ylabel("Key Error Rate")
    plt.legend()
    plt.semilogy(base=10)
    
    #tikzplotlib.clean_figure()
    #tikzplotlib.save("figure_to_be_renammed.tex")

    plt.show()

def plot_KER_BEST_PIN(mod="CB",temp=30):
    proba = np.linspace(0,1,10**4)

    for label in ["SHA256"]:
        cpath = f"../../puf-no-hda/HDA_RESULTS/{mod}/{temp}C/{label}/"
        for i in [1,2,4,8]:

            ker = np.load(cpath+"KER.npy")
            j = len(ker) // i
            print(ker)
            ker = np.min(ker[0:i*j].reshape(j,i),axis=1)
            ker_cdf = cdf_ker(1-ker)
            plt.plot(1-ker_cdf,1-proba,label=label + f" m = {i}")
            print(f"Average KER {label}  {i} : ", np.mean(ker) )
    plt.grid()
    plt.xlabel("Survival Distribution Function")
    plt.ylabel("Key Error Rate")
    plt.legend()
    plt.semilogy(base=10)
    
    tikzplotlib.clean_figure()
    tikzplotlib.save("figure_to_be_renammed.tex")

    plt.show()

def plot_BER(mod="CB",temp=30):
    proba = np.linspace(0,.1,10**5)

    for label in Labels:
        cpath = f"../../puf-no-hda/HDA_RESULTS/{mod}/{temp}C/{label}/"
        ber = np.load(cpath+"BER.npy")
        ber_cdf = cdf_ber(ber)
        plt.plot(ber_cdf,proba,label=label)
    plt.grid()
    plt.xlabel("Survival Distribution Function")
    plt.ylabel("Key Error Rate")
    plt.legend()
    plt.semilogy(base=10)
    
    tikzplotlib.clean_figure()
    tikzplotlib.save("figure_to_be_renammed.tex")

    plt.show()



def plot_BER_BEST_PIN(mod="CB",temp=30):
    proba = np.linspace(0,.1,10**5)

    for label in ["SECDED_ML"]:
        cpath = f"../../puf-no-hda/HDA_RESULTS/{mod}/{temp}C/{label}/"
        for i in [1,2,4,8]:
            ber = np.load(cpath+"BER.npy")
            ber = np.min(ber.reshape(-1,i),axis=1)
            ber_cdf = cdf_ber(ber)
            plt.plot(ber_cdf,proba,label=label+f" m = {i}")
    plt.grid()
    plt.xlabel("Survival Distribution Function")
    plt.ylabel("Key Error Rate")
    plt.legend()
    plt.semilogy(base=10)
    #if save:
    #    plt.savefig(name+".pdf",format="pdf",dpi=600)

    tikzplotlib.clean_figure()
    tikzplotlib.save("figure_to_be_renammed.tex")

    plt.show()



import numpy as np 
import pandas as pd 
import os

def load(f):
    return pd.read_csv(f)

def extract_rep(df,rep):
    return df[df["repetitions"]==rep].drop(["repetitions"],axis="columns")


"""
Input: This functions takes as input an window size, a pin setting and an index.
 - windowsize : int between 2**11 and 2 ** 20  -9 possibilities-
 - pin_setting : int between 0 and 7  - 8 possibilities -
 - index : int between 0 and  63
Output: This return the lines of the dataframe that fits with these parameters. For the res^ponse_out_1k file there should be 10**3 lines.
"""
def get(winsize=2**15,pin_setting=0,index=0):
    return df[(df["winsize"]==winsize) * (df["pin_setting"]==pin_setting) * (df["index"]==index)]

def read_temp(temp :int,board :int,puf :int,pins :int,mod :str) -> pd.core.frame.DataFrame:
    prefix = "../../MESURE_LPUF/MESURE_TEMPERATURE/"
    suffix = f"BOARD{board}_{temp}C_{mod}/BOARD{board}_{mod}_puf_{puf}_pins_{pins}.csv"
    path = prefix+suffix
    print("Loading : " + path + " ...")
    return load(path)

path_to_temp = "../../MESURE_LPUF/MESURE_TEMPERATURE/"
def read(temp=30,mod="CB"):
    L = []
    for board in range(1,6):
        path_board = path_to_temp+f"BOARD{board}_{temp}C_{mod}"
        files = os.listdir(path_board)
        for f in files:
            L.append(path_board+"/"+f)
    return L

#################################################################
    # List paths measurments for CB and different temperatures 
#################################################################
L30C_CB = read(temp=30,mod="CB")
L40C_CB = read(temp=40,mod="CB")
L50C_CB = read(temp=50,mod="CB")
L60C_CB = read(temp=60,mod="CB")

#################################################################
    # List paths measurments for SQRT and different temperatures 
#################################################################
L30C_SQRT = read(temp=30,mod="SQRT")
L40C_SQRT = read(temp=40,mod="SQRT")
L50C_SQRT = read(temp=50,mod="SQRT")
L60C_SQRT = read(temp=60,mod="SQRT")

#################################################################################
################### OLD MEASURMENTS FILES #######################################
#################################################################################
fc  = "../../lpuf_analysis/measurement/data/response_out_1k.csv"
fr  = "../../lpuf_analysis/measurement/data/fr_puf_response_out_1k.csv"
fc_new  = "../../lpuf_analysis/measurement/data/classic_puf_response_out_1k_04_10.csv"
fr_new  = "../../lpuf_analysis/measurement/data/fr_puf_response_out_1k_04_08.csv"
fr_new_new = "../../lpuf_analysis/measurement/data/fr_puf_response_out_1k_04_27.csv"
fr_rep = "../../lpuf_analysis/measurement/data/fr_puf_reps_2-4-8-16_1k.csv"


etalon =  "../soft/measurement/LPUF_AUTOREPEAT_t100_rmax65536_.csv"
etalon2 = "../soft/measurement/fr_puf_response_winsize_1024_nmeas_50.csv"
etalon3 = "../soft/test_test_test"


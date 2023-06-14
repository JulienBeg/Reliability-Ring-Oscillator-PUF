import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hashlib
from hashlib import sha256

from puf_dataset import *

#df = load(AutoRepeat[0])
df = load(L30C_SQRT[0])

tuples_5_sorted = np.load("5tuplesSorted.npy")
tuples_4_sorted = np.load("4tuplesSorted.npy")
tuples_3_sorted = np.load("3tuplesSorted.npy")
tuples_2_sorted = np.load("2tuplesSorted.npy")
tuples_1_sorted = np.array([[i] for i in range(64)])

"""
Compute the BER auf the Autorepeat PUF design.
"""
def enrolled_key(df,Nmeasures=1000):

    #Drop Useless Columns
    dfe = df.drop(['winsize','pin_setting','frequency','puf_num'],axis='columns')

    #Split the measures into two dataframes
    # - enrolment with 5% of the data is used for the key enrolment
    # - evaluation with 95% of the data is used to evaluate the ber based on the enrolled key
    Nevaluation = int(.95*Nmeasures)
    Nenrolement = Nmeasures - Nevaluation

    df_enrolment  = dfe[(dfe["run_idx"]>=Nevaluation)]["delta_freq"].to_numpy().reshape(Nenrolement,64).sum(axis=0)

    #Compute the enrolled key
    enrolled_key = (df_enrolment > 0)
    return enrolled_key

"""
Convert arry of bits to integer
"""
def bits_to_int(A):
    I = np.arange(len(A))
    return np.sum(A * 2 ** I)

"""
Flip the bits in the list in the given array
"""
def flips(Array,I):
    for i in I:
        Array[i] = not Array[i]

"""
Compute the BER auf the Autorepeat PUF design.
"""
def ber2(df,Nmeasures=1000):

    #Drop Useless Columns
    dfe = df.drop(['winsize','pin_setting','frequency','puf_num'],axis='columns')

    #Split the measures into two dataframes
    # - enrolment with 5% of the data is used for the key enrolment
    # - evaluation with 95% of the data is used to evaluate the ber based on the enrolled key
    Nevaluation = int(.95*Nmeasures)
    Nenrolement = Nmeasures - Nevaluation

    df_enrolment  = dfe[(dfe["run_idx"]>=Nevaluation)]["delta_freq"].to_numpy().reshape(Nenrolement,64).sum(axis=0)
    df_evaluation = dfe[(dfe["run_idx"]< Nevaluation)]["delta_freq"].to_numpy().reshape(Nevaluation,64)

    #Compute the enrolled key
    enrolled_key = (df_enrolment > 0)

    #Compute the reproduces keys and compare with the enrolled one
    test = ((df_evaluation > 0) != enrolled_key)
    ker = ((df_evaluation > 0) == enrolled_key).all(axis=1)

    return test.mean(),test.std(),ker.mean(),ker.std()

"""
Compute the BER of the Autorepeat PUF design when it is endowed with N parity check bits.
In other words there is N parrallel parity check codes of lenght 64/N.
It is assumed that N divides 64.

The decoding procedure is done as follows: If a parity does not match then the least reliable bits linked with this parity is flipped.
"""
def ber_with_parities(df,Nmeasures=1000,N=2):

    #Drop Useless Columns
    dfe = df.drop(['winsize','pin_setting','frequency','puf_num'],axis='columns')

    #Split the measures into two dataframes
    # - enrolment with 5% of the data is used for the key enrolment
    # - evaluation with 95% of the data is used to evaluate the ber based on the enrolled key
    Nevaluation = int(.95*Nmeasures)
    Nenrolement = Nmeasures - Nevaluation

    df_enrolment  = dfe[(dfe["run_idx"]>=Nevaluation)]["delta_freq"].to_numpy().reshape(Nenrolement,64).sum(axis=0)
    df_evaluation = dfe[(dfe["run_idx"]< Nevaluation)]["delta_freq"].to_numpy().reshape(Nevaluation,64)
    t_evaluation  = dfe[(dfe["run_idx"]< Nevaluation)]["number_measurments"].to_numpy().reshape(Nevaluation,64)

    #Compute the enrolled key
    enrolled_key = (df_enrolment > 0)

    width = 64//N
    parities = np.sum(enrolled_key.reshape(N,width),axis=1)% 2

    #Compute the reproduced keys
    reproduced_keys = (df_evaluation >0)

    decoding_err = 0
    detected_err = 0

    test,ker=[],[]
    for i in range(Nevaluation):

        rep = reproduced_keys[i]

        parities_rep = np.sum(rep.reshape(N,width),axis=1) % 2

        for j in range(N):
            if parities_rep[j] != parities[j]:
                detected_err += 1

                error_location = j*width + np.argmin( np.abs(  df_evaluation[i]  / np.sqrt( t_evaluation[i]) )[j*width:(j+1)*width])
                #print(len( np.abs(  df_evaluation[i]  / np.sqrt( t_evaluation[i]) )[j*width:(j+1)*width]))

                if rep[error_location] == enrolled_key[error_location]:
                    decoding_err += 1

                rep[error_location] = (1+rep[error_location])%2

        test.append( rep != enrolled_key)
        ker.append( (rep == enrolled_key).all() )

    test = np.array(test)
    ker = np.array(ker)
    return test.mean(),test.std(),ker.mean(),ker.std()

"""
Modified Hamming :
    1 ) We consider a [127,120,3]_2 Hamming code.
    2 ) We puncter it to a code of length 64 (and unkown dimension)
    3 ) We extend the code with a parity check code equation
We obtain a modified Hamming code (SECDED) code of length 64 and 8 parity equations.
"""
# This is the parity check matrix of the Hamming code
Hhamming = np.array([ list(np.binary_repr(i,7)) for i in range(1,2**7)],dtype=np.uint8)
# We truncate the code to a length 64
Hsecded = Hhamming.T[:,0:64]
def SECDED_ALG(df,Nmeasures=1000):

    #Split the measures into two dataframes
    # - enrolment with 5% of the data is used for the key enrolment
    # - evaluation with 95% of the data is used to evaluate the ber based on the enrolled key
    Nevaluation = int(.95*Nmeasures)
    Nenrolement = Nmeasures - Nevaluation

    #Drop Useless Columns
    dfe = df.drop(['winsize','pin_setting','frequency','puf_num'],axis='columns')

    df_enrolment  = dfe[(dfe["run_idx"]>=Nevaluation)]["delta_freq"].to_numpy().reshape(Nenrolement,64).sum(axis=0)
    df_evaluation = dfe[(dfe["run_idx"]< Nevaluation)]["delta_freq"].to_numpy().reshape(Nevaluation,64)
    t_evaluation  = dfe[(dfe["run_idx"]< Nevaluation)]["number_measurments"].to_numpy().reshape(Nevaluation,64)

    #Compute the enrolled key
    enrolled_key = (df_enrolment > 0)

    #Compute Syndrome of enrolled key
    SS = (np.matmul(Hsecded,enrolled_key)%2).astype(bool)
    ExtraParity = np.sum(enrolled_key)%2

    #Compute the reproduced keys
    reproduced_keys = (df_evaluation >0)

    test,ker=[],[]
    for i in range(Nevaluation):

        # We consider the i-th experiments
        rep = np.copy(reproduced_keys[i])

        # We compute the a posterioti least reliable bits
        si = np.abs( df_evaluation[i]/np.sqrt(t_evaluation[i]) )
        least_reliables = np.argsort(si)

        SStilde = (np.matmul(Hsecded,rep)%2).astype(bool)
        ExtraParityTilde = np.sum(rep)%2
        syndrome = SS ^ SStilde

        if syndrome.any():

            if ExtraParityTilde == ExtraParity:
                #There is an even number of errors
                #We treat it as a 2 errors patterns

                j = 0
                ok = False

                while not ok:

                    #We identify the j-th least reliable bit
                    err = least_reliables[j]
                    #We flip this bit
                    flips(rep,[err])

                    #We compute the new syndrome
                    SStilde = (np.matmul(Hsecded,rep)%2).astype(bool)
                    syndrome = SS ^ SStilde

                    #Compute the error location
                    error_position = bits_to_int(np.flip(syndrome))-1

                    #We verify we went down to a 1 error pattern
                    if error_position < 64:

                        #Flip the bit identified with the syndrome
                        flips(rep,[error_position])
                        #Stop the loop
                        ok=True
                    else:
                        #This corresponds to a three error pattern.
                        j+=1
                        flips(rep,[err])
                        #rep[err] = not rep[err]

            else:
                #There is an odd number of error

                #Compute the error location
                error_position = bits_to_int(np.flip(syndrome))-1

                if error_position < 64:
                    #This corresponds to a 1 error pattern
                    flips(rep,[error_position])
                else:
                    # This coresponds to a 3 errors pattern (at least)
                    # We try to reduce it to a 1 error patter by sequentialy fliping least
                    # reliable bits (by increasing L1 norm)
                    smax,sum_ij,ok = 10,0,False
                    while (sum_ij < smax) and (not ok):
                        sum_ij += 1
                        for i in range(sum_ij//2):
                            j = sum_ij  - i
                            #Flip these least reliable bits
                            flips(rep,[least_reliables[i],least_reliables[j]])

                            #Compute the syndrome
                            SStilde = (np.matmul(Hsecded,rep)%2).astype(bool)
                            syndrome = SS ^ SStilde

                            #Compute the error location
                            error_position = bits_to_int(np.flip(syndrome))-1

                            #We verify we went down to a 1 error pattern
                            if error_position < 64:
                                #Flip the bit identified with the syndrome
                                flips(rep,[error_position])
                                #Stop the loop
                                ok=True
                                break
                            else:
                                #Unflip
                                flips(rep,[least_reliables[i],least_reliables[j]])

        test.append( rep != enrolled_key)
        ker.append( (rep == enrolled_key).all() )
    test = np.array(test)
    ker = np.array(ker)
    return test.mean(),test.std(),ker.mean(),ker.std()

def SECDED_ML(df,Nmeasures=1000):

    #Split the measures into two dataframes
    # - enrolment with 5% of the data is used for the key enrolment
    # - evaluation with 95% of the data is used to evaluate the ber based on the enrolled key
    Nevaluation = int(.95*Nmeasures)
    Nenrolement = Nmeasures - Nevaluation

    #Drop Useless Columns
    dfe = df.drop(['winsize','pin_setting','frequency','puf_num'],axis='columns')

    df_enrolment  = dfe[(dfe["run_idx"]>=Nevaluation)]["delta_freq"].to_numpy().reshape(Nenrolement,64).sum(axis=0)
    df_evaluation = dfe[(dfe["run_idx"]< Nevaluation)]["delta_freq"].to_numpy().reshape(Nevaluation,64)
    t_evaluation  = dfe[(dfe["run_idx"]< Nevaluation)]["number_measurments"].to_numpy().reshape(Nevaluation,64)

    #Compute the enrolled key
    enrolled_key = (df_enrolment > 0)

    #Compute Syndrome of enrolled key
    SS = (np.matmul(Hsecded,enrolled_key)%2).astype(bool)
    ExtraParity = np.sum(enrolled_key)%2

    #Compute the reproduced keys
    reproduced_keys = (df_evaluation >0)

    test,ker=[],[]
    for i in range(Nevaluation):

        # We consider the i-th experiments
        rep = np.copy(reproduced_keys[i])

        # We compute the a posterioti least reliable bits
        si = np.abs( df_evaluation[i]/np.sqrt(t_evaluation[i]) )
        least_reliables = np.argsort(si)

        SStilde = (np.matmul(Hsecded,rep)%2).astype(bool)
        ExtraParityTilde = np.sum(rep)%2
        syndrome = SS ^ SStilde

        """
        Auxiliary functions that tries the key candidates modified a the bit index given in a tuple
        for all the tuples in a list. It stopps when it finds a candidate.
        """
        def try_tuples(tuples):
            for tuple in tuples:
                #Flip these least reliable bits
                flips(rep,least_reliables[tuple])

                SStilde = (np.matmul(Hsecded,rep)%2).astype(bool)
                syndrome = SS ^ SStilde

                if not syndrome.any():
                    return True
                else:
                    flips(rep,least_reliables[tuple])
            return False

        ok = not (syndrome.any() or (ExtraParityTilde != ExtraParity))
        if ExtraParityTilde != ExtraParity:#There is an odd number of errors
            #We try 1 error patterns
            if not ok:
                ok = try_tuples(tuples_1_sorted[0:10])
            #We try 3 errors patterns
            if not ok:
                ok = try_tuples(tuples_3_sorted[0:40])
            if not ok:
                print("Decoder failed ! ")
        else:#There is an even number of error
            #We try 2 errors patterns
            if not ok:
                ok = try_tuples(tuples_2_sorted[0:20])
            #We try 4 errors patterns
            if not ok:
                ok = try_tuples(tuples_4_sorted[0:40])
            if not ok:
                print("Decoder failed ! ")
        test.append( rep != enrolled_key)
        ker.append( (rep == enrolled_key).all() )
    test = np.array(test)
    ker = np.array(ker)
    return test.mean(),test.std(),ker.mean(),ker.std()

"""
HASH (SHA-256)
    1) We assume that the hash of the key is stored as helper data.
    2) We try to brute force the hash using the puf observation as a trapdoor.
"""
def HASH(df,Nmeasures=1000):

    #Drop Useless Columns
    dfe = df.drop(['winsize','pin_setting','frequency','puf_num'],axis='columns')

    #Split the measures into two dataframes
    # - enrolment with 5% of the data is used for the key enrolment
    # - evaluation with 95% of the data is used to evaluate the ber based on the enrolled key
    Nevaluation = int(.95*Nmeasures)
    Nenrolement = Nmeasures - Nevaluation

    df_enrolment  = dfe[(dfe["run_idx"]>=Nevaluation)]["delta_freq"].to_numpy().reshape(Nenrolement,64).sum(axis=0)
    df_evaluation = dfe[(dfe["run_idx"]< Nevaluation)]["delta_freq"].to_numpy().reshape(Nevaluation,64)
    t_evaluation  = dfe[(dfe["run_idx"]< Nevaluation)]["number_measurments"].to_numpy().reshape(Nevaluation,64)

    #Compute the enrolled key
    enrolled_key = (df_enrolment > 0)

    m = hashlib.sha256()
    m.update(enrolled_key.astype("bool").tobytes())
    hash = m.digest()

    #Compute the reproduced keys
    reproduced_keys = (df_evaluation >0)

    decoding_err = 0
    detected_err = 0

    test,ker=[],[]
    for i in range(Nevaluation):

        rep = reproduced_keys[i]
        si = np.abs( df_evaluation[i]/np.sqrt(t_evaluation[i]) )
        #n_unstable = np.sum( si < 49)

        least_reliables = np.argsort(si)

        m = hashlib.sha256()
        m.update(rep.astype("bool").tobytes())
        hash_rep = m.digest()


        """
        Auxiliary functions that tries the key candidates modified a the bit index given in a tuple
        for all the tuples in a list. It stopps when it finds a candidate.
        """
        def try_tuples(tuples):
            for tuple in tuples:
                #Flip these least reliable bits
                flips(rep,least_reliables[tuple])

                m = hashlib.sha256()
                m.update(rep.astype("bool").tobytes())
                hash_rep = m.digest()
                if hash == hash_rep:
                    return hash_rep
                else:
                    flips(rep,least_reliables[tuple])
            return hash_rep

        #We first try 1 error patterns
        if not (hash == hash_rep):
            hash_rep = try_tuples(tuples_1_sorted[0:30])

        #We try two errors patterns
        if not (hash == hash_rep):
            hash_rep = try_tuples(tuples_2_sorted[0:100])

        #We try three errors pattersns
        if not(hash == hash_rep):
            hash_rep = try_tuples(tuples_3_sorted[0:100])

        #We try four error patterns
        if not(hash == hash_rep):
            hash_rep = try_tuples(tuples_4_sorted[0:100])

        #We try five error patterns
        if not(hash == hash_rep):
            hash_rep = try_tuples(tuples_5_sorted[0:100])

        #At this point we give up (we could try more).
        if not (hash_rep == hash):
            print("Decoder failed !")
            print("The number of error to be corrected was ", np.sum( enrolled_key.astype("bool") ^ rep.astype("bool") ))

        test.append( rep != enrolled_key)
        ker.append( (rep == enrolled_key).all() )
    test = np.array(test)
    ker = np.array(ker)
    return test.mean(),test.std(),ker.mean(),ker.std()


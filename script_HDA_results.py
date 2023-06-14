from Autorepeat import * 

Algs = [ber2, lambda x: ber_with_parities(x,N=1),lambda x: ber_with_parities(x,N=2),lambda x: ber_with_parities(x,N=4),lambda x: ber_with_parities(x,N=8),SECDED_ALG,SECDED_ML,HASH]
Labels = ["WITHOUT_HDA","1PCK","2PCK","4PCK","8PCK","SECDED_ALG","SECDED_ML","SHA256"]
MOD = ["CB","SQRT"] 
TEMP = [30,40,50,60]
BOARDS = 1+np.arange(5)
PUFS = np.arange(16)
PINS = np.arange(8)

for mod in MOD:
    for temp in TEMP:
        it=-1
        for HDA in Algs:
            
            it+=1
            alg_label = Labels[it] 

            cpath = f"HDA_RESULTS/{mod}/{temp}C/{alg_label}/"
            if not os.path.exists(cpath):
                os.makedirs(cpath)

            KER,BER = [],[]
            for board in BOARDS:
                for puf in PUFS:
                    for pins in PINS:

                        df = read_temp(temp,board,puf,pins,mod)
                        _,ber,_,ker = HDA(df)
                        KER.append(ker)
                        BER.append(ber)
            
            print(KER)
            print(BER)
            print("Saving Results : " + cpath + "...")
            with open(cpath+"KER.npy","wb") as f:
                np.save(f,KER)
            with open(cpath+"BER.npy","wb") as f:
                np.save(f,BER)
                    

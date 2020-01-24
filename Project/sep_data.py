if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import os
    import datetime
    import sys 
    import csv
    from sklearn import svm


    filename = "data.csv"
    
    df = pd.read_csv(filename)
    leng = (len(df))
    batch_size = int(leng / 20)

    for i in range(20):
        out_file_name = "data_batch" + str(i) + ".csv"
        if i < 19:
            df_out = df.iloc[batch_size*i : batch_size*(i+1)]
            df_out.to_csv("./"+ out_file_name)
        df_out = df.iloc[batch_size*i : ]
        df_out.to_csv("./"+ out_file_name)

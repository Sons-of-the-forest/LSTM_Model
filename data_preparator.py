import pandas as pd
import numpy as np
import pickle
import os
import re

class DataPreparator:
    def __init__(self, folder_path='.\dataRaw'):
        self.folder_path = folder_path
        self.words = []
        self.df_list = []
        self.df_property = []
        self.data = None

    def prepare_data(self):    
        self.get_words()
        self.concat_data()
        self.save_data()
        
    def get_words(self):
        self.words = []
        for path in os.walk('.\dataRaw'):
            try :
                self.words.append(path[0].split('\\')[2])
                # print("Added folder "+ path[0].split('\\')[2] +" into list")
            except:
                pass
                # print("Except path: ", path[0])

    def concat_data(self):
        self.df_list = []
        self.df_property = []

        for word in self.words:
            path = "{}\{}".format(self.folder_path, word)
            files = []
            for file_path in os.walk(path):
                files = file_path[2]
            
            for file in files:
                file_path = path + "\{}".format(file)
                df = pd.read_csv(file_path)
                df = df[['First Byte', ' Second Byte', ' Brainwave Value']]
                df.columns = ['v_0', 'v_1', 'v_b']
                df['repetition'] = int(re.findall(r'\d+', file)[0])
                df['label'] = word
                self.df_list.append(df)
                self.df_property.append((word, int(re.findall(r'\d+', file)[0])))

        self.data = pd.concat(self.df_list, axis=0)
    
    def save_data(self, path='.\data\prepared_data.csv'):
        self.data.to_csv(path)
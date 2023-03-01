import pandas as pd
import numpy as np

from data_preparator import DataPreparator

class DataPreprocessor:
    def __init__(self):
        data_preparator = DataPreparator()
        data_preparator.prepare_data()
        
        self.words = data_preparator.words
        self.df_list = data_preparator.df_list
        self.df_property = data_preparator.df_property
        self.data = data_preparator.data


    def preprocess(self):
        self.pad_word()
        self.word_encoding()
        self.save_data()

    def pad_word(self):
        self.word_length = max(len(df) for df in self.df_list)
        temp_list = []
        for df in self.df_list:
            df = df.reindex(range(self.word_length)).fillna(df.min())
            temp_list.append(df)
        self.df_list = temp_list
        self.data = pd.concat(self.df_list, axis=0)
    
    def word_encoding(self):
        one_hot = pd.get_dummies(self.data['label']).values
        one_hot_combined = np.concatenate(one_hot, axis=0).reshape(len(self.data), -1)
        self.data['one_hot_label'] = list(one_hot_combined)

    def save_data(self, path='.\data\preprocessed_data.csv'):
        self.data.to_csv(path)
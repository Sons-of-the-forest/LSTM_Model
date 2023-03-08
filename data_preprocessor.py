import pandas as pd
import numpy as np
# import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessor:
    def __init__(self, df_list=None, df_property=None, words=None, scaler='standard'):
        self.df_list = df_list
        self.words = words
        if scaler == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        self.df_property = df_property
        self.data = None
        self.word_length = 0
        self.features = self.df_list[0].columns.to_list()
        self.features.remove("label")
        self.features.remove("repetition")
        self.X = []
        self.y = []

    def preprocess(self):
        # self.fourier_transform()
        self.pad_word()
        self.select_features()
        # self.scale_data()
        self.encode_label()
        self.get_datapoints()
        return self.X, self.y

    def select_features(self, features=['v_b']):
        new_df_list = []
        for df in self.df_list:
            new_df_list.append(df[features])
        self.df_list = new_df_list
        self.features = features

    def pad_word(self):

        new_df_list = []
        new_df_property = []
        print("Padding words...")
        print("Original df_list contains: {} dataframes".format(len(self.df_list)))
        for i, df in enumerate(self.df_list):
            if len(df) >= 1500:
                print("Remove {} with length {}".format(self.df_property[i], len(df)))
            else:
                new_df_list.append(df)
                new_df_property.append(self.df_property[i])
        self.df_list = new_df_list
        self.df_property = new_df_property
        print("Prunned df_list contains: {} dataframes".format(len(self.df_list)))
        
        # length = [len(df) for df in self.df_list]
        self.word_length = 1500
        # print("Max word length:", self.df_property[np.argmax(np.array(length))])
        temp_list = []
        for df in self.df_list:
            df = df.reindex(range(self.word_length)).fillna(df.min())
            # df[['v_0', 'v_1', 'v_b']] = df[['v_0', 'v_1', 'v_b']].apply(pd.to_numeric, errors='coerce')
            temp_list.append(df)
        self.df_list = temp_list
        self.data = pd.concat(self.df_list, axis=0)

        # string_series = self.data[['v_0', 'v_1', 'v_b']].select_dtypes(include=['object']).stack()
        # is_string = string_series.apply(lambda x: isinstance(x, str))
        # print(string_series[is_string])
        # has_non_numeric = self.data[['v_0', 'v_1', 'v_b']].isna().any(axis=1)
        # print(self.data[has_non_numeric])
    
    def scale_data(self):
        # self.features = self.data.columns.remove("label").remove("repetition")
        self.data[self.features] = self.scaler.fit_transform(self.data[self.features])
        temp_list = []
        for df in self.df_list:
            # df[['v_0', 'v_1', 'v_b']] = df[['v_0', 'v_1', 'v_b']].apply(pd.to_numeric, errors='coerce')

            df[self.features] = self.scaler.transform(df[self.features])
            temp_list.append(df)
        self.df_list = temp_list


    def encode_label(self):
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

        one_hot_encoder.fit(np.array(self.words).reshape(-1, 1))
        label = [self.df_property[i][0] for i in range(len(self.df_property))]
        self.y = one_hot_encoder.transform(np.array(label).reshape(-1, 1))

    def get_datapoints(self):
        self.X = []
        for df in self.df_list:
            self.X.append(df[self.features].values.tolist())
        self.X = np.array(self.X)

    def fourier_transform(self):
        temp_list = []
        for df in self.df_list:
            for column in ['v_0', 'v_1', 'v_b']:
                fft = np.fft.fft(df[column])
                fft_df = {
                    column+ '_real' : fft.real,
                    column+ '_imaginary' : fft.imag,
                    column+ '_amplitude' : np.abs(fft),
                    column+ '_phase' : np.angle(fft)
                }
                fft_df = pd.DataFrame(fft_df)
                df = pd.concat([df, fft_df], axis=1)
            temp_list.append(df)
        
        self.df_list = temp_list
        self.data = pd.concat(self.df_list, axis=0)
        self.features = self.data.columns.to_list()
        self.features.remove("label")
        self.features.remove("repetition")
                


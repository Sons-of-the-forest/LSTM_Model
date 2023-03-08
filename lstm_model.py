# import tensorflow as tf
# print(tf.test.gpu_device_name())

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

class LstmModel:
    def __init__(self, word_length=1000, input_dim=3, num_class=100):
        self.model = Sequential()

        self.word_length = word_length
        self.input_dim = input_dim
        self.num_class = num_class

        self.early_stopping = EarlyStopping(monitor='val_loss', verbose=2, mode=min, patience=50)
        self.history = None
        self.define_model()

    def define_model(self):
        # self.model.add(Dense(units=self.input_dim, activation='softmax'))
        self.model.add(LSTM(units=256, input_shape=(self.word_length, self.input_dim), return_sequences=True))
        # self.model.add(LSTM(units=256, input_shape=(self.word_length, self.input_dim), return_sequences=True))
        # self.model.add(LSTM(units=256, input_shape=(self.word_length, self.input_dim), return_sequences=True))
        # self.model.add(LSTM(units=256, input_shape=(self.word_length, self.input_dim), return_sequences=True))
        # self.model.add(LSTM(units=256, input_shape=(self.word_length, self.input_dim), return_sequences=True))
        self.model.add(LSTM(units=128, input_shape=(self.word_length, self.input_dim)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(units=self.num_class, activation='softmax'))

        adam = Adam(lr=0.001)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=adam,
            metrics=['accuracy'])
        
        # self.print_model()
    
    def print_model(self):
        print(self.model.summary())

    def fit(self, num_epochs=20, batch_size=16):
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=num_epochs, batch_size=batch_size, callbacks=[self.early_stopping])

    
    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print('Loss : {}'.format(loss))
        print('Accuracy : {}'.format(accuracy))
    
    def split_dataset(self, X, y, test_size=0.2, val_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=val_size)
        # return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
    
    # def train_model(self, X, y):
    #     self.split_dataset(self, X, y)
    #     self.fit()
    #     self.evaluate()
    #     with open('.\model', 'wb') as f:
    #         pickle.dumps(self, f):
        

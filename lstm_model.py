# import tensorflow as tf
# print(tf.test.gpu_device_name())
import parameter as para

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Parameters
word_max_length = para.WORD_LENGTH
input_dim = para.INPUT_DIM
num_class = para.NUM_CLASS


# Define model
model =  Sequential()
model.add(LSTM(64, input_shape=(word_max_length, input_dim), return_sequences=True))
model.add(Dense(num_class, activation='softmax'))

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy', 'f-score']
)

# Train model
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=batch_size, callbacks=[early_stopping])

loss, accuracy, fscore = model.evaluate(x_test, y_test)
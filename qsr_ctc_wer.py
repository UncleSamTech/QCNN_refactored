
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, BatchNormalization, Conv1D, GRU, TimeDistributed, Activation, Dense, Flatten
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy
import os
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from helper_q_tool import plot_acc_loss,plot_acc_loss2
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import string
from models import build_asr_model
import json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time as ti
data_ix = ti.strftime("%m%d_%H%M")
labels = ['hey_android',  'hey_snapdragon', 'hi_lumina', 'hi_galaxy']

characters = string.ascii_lowercase # set(char for label in labels for char in label)

# Mapping characters to integers
char_to_num = L.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), num_oov_indices=0, mask_token=None
)

# Mapping integers back to original characters
num_to_char = L.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

# test_x = tf.random.uniform([4, 60, 126, 1])
# test_y = labels 

b_size = 64
MAX_word_length = 2
SAVE_PATH = "data_quantum/asr_set/"
load_asr_data = True


def get_asr_data(y_valid, y_train, x_valid, x_train, q_valid, q_train):
    char_y_tr = []
    char_y_val = []
    new_x_tr = []
    new_x_val = []
    new_q_tr = []
    new_q_val = []
    
    for idx, y in enumerate(y_valid):
         if labels_10[np.argmax(y)] in labels:
             char_y_val.append(labels_10[np.argmax(y)])
             new_x_val.append(x_valid[idx])
             new_q_val.append(q_valid[idx])

    for idx, y in enumerate(y_train):
         if labels_10[np.argmax(y)] in labels:
             char_y_tr.append(labels_10[np.argmax(y)])
             new_x_tr.append(x_train[idx])
             new_q_tr.append(q_train[idx])

    return char_y_tr, char_y_val, new_x_tr, new_x_val, new_q_tr, new_q_val

if load_asr_data == True:
    print("Load Data")
    new_x_tr = np.load(SAVE_PATH + "asr_x_tr.npy")
    new_x_val = np.load(SAVE_PATH + "asr_x_val.npy")
    new_q_tr = np.load(SAVE_PATH + "asr_q_tr.npy")
    new_q_val = np.load(SAVE_PATH + "asr_q_val.npy")
    with open(SAVE_PATH + "char_y_val.json", 'r') as f:
        char_y_val = json.load(f)
    with open(SAVE_PATH + "char_y_tr.json", 'r') as f:
        char_y_tr = json.load(f)    
else:
    print("Please Proc. your features")
    exit()

print("-- Validation Size: ", np.array(char_y_val).shape, np.array(new_x_val).shape, np.array(new_q_val).shape)
print("-- Training Size: ", np.array(char_y_tr).shape, np.array(new_x_tr).shape, np.array(new_q_tr).shape)


# Get the model
model = build_asr_model(30, 63, 4) # 60 126 1
model.summary()


def encode_single_sample(img, label):
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    # img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return {"speech": img, "label": label}

print("=== Making CTC input dataset ...")

train_dataset = tf.data.Dataset.from_tensor_slices((new_q_tr, char_y_tr))
train_dataset = (
    train_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(b_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((new_q_val, char_y_val))
validation_dataset = (
    validation_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(b_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

epochs = 50
early_stopping_patience = 10
# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

history = model.fit(train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping],)

plot_acc_loss2(history,data_ix)
# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name="speech").input, model.get_layer(name="dense2").output
)

# prediction_model.summary()

# A utility function to decode the output of the network
def decode_batch_predictions(pred, max_length=MAX_word_length):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

val_port = 3
pred_texts = []

for batch in validation_dataset.take(val_port):
    batch_speech = batch["speech"]
    preds = prediction_model.predict(batch_speech)
    pred_texts.append(decode_batch_predictions(preds))

import itertools
cor_idx = 0
pred_texts = list(itertools.chain.from_iterable(pred_texts))

for idx, word in enumerate(char_y_val[0:b_size*val_port]):
    if word != pred_texts[idx]:
        cor_idx += 1
#    else:
#        if len(word) == len(pred_texts[idx]):
#           pass CER
print(pred_texts)
print("=== WER:", 100*cor_idx/len(pred_texts), " %")
model.save('checkpoints/' + 'asr_ctc_demo.hdf5')

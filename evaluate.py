import os
import csv
from utils import *
from sklearn.model_selection import train_test_split
from transformers import *
from D2SBERT import *
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])

max_len = 512

code = ['401.9', '38.93', '428.0', '427.31', '414.01', '96.04', '96.6', '584.9', '250.00', '96.71', '272.4', '518.81',
        '99.04', '39.61', '599.0', '530.81', '96.72', '272.0', '285.9', '88.56', '244.9', '486', '38.91', '285.1',
        '36.15', '276.2', '496', '99.15', '995.92', 'V58.61', '507.0', '038.9', '88.72', '585.9', '403.90', '311',
        '305.1', '37.22', '412', '33.24', '39.95', '287.5', '410.71', '276.1', 'V45.81', '424.0', '45.13', 'V15.82',
        '511.9', '37.23']

file = open('./notes_labeled.csv', 'r')

rdr = csv.reader(file)

texts, labels = data_preprocessing(rdr, code)

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1-mnli")

train_inputs, train_data_labels = model_input(X_train, y_train, tokenizer, max_len)
dev_inputs, dev_data_labels = model_input(X_dev, y_dev, tokenizer, max_len)
test_inputs, test_data_labels = model_input(X_test, y_test, tokenizer, max_len)

with strategy.scope():

    cls_model = D2SBERT_Model(model_name='dmis-lab/biobert-base-cased-v1.1-mnli', dir_path='bert_ckpt', num_class=50)
    F1_macro = tfa.metrics.F1Score(num_classes=50, average='macro', threshold=0.5, name='f1_macro')
    F1_micro = tfa.metrics.F1Score(num_classes=50, average='micro', threshold=0.5, name='f1_micro')

    optimizer = tf.keras.optimizers.Adam(1e-5)
    loss = tf.keras.losses.BinaryCrossentropy()
    metric = tf.keras.metrics.BinaryAccuracy()

    cls_model.compile(optimizer=optimizer, loss=loss, metrics=[metric, F1_macro, F1_micro])

EarlyStopping = EarlyStopping(monitor='val_f1_macro', verbose=1, min_delta=0.0001, patience=4, mode='max',
                              restore_best_weights=True)

checkpoint_path = os.path.join('./', '', 'weights.h5')
checkpoint_dir = os.path.dirname(checkpoint_path)

if os.path.exists(checkpoint_dir):
    print("{} -- Folder already exists \n".format(checkpoint_dir))
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("{} -- Folder create complete \n".format(checkpoint_dir))

cp_callback = ModelCheckpoint(
    checkpoint_path, monitor='val_f1_macro', mode='max', verbose=1, save_best_only=True, save_weights_only=True)

cls_model.fit(train_inputs, train_data_labels, epochs=50, batch_size=4, validation_data=(dev_inputs, dev_data_labels),
              callbacks=[EarlyStopping, cp_callback])

test_loss, test_acc, test_macro, test_micro = cls_model.evaluate(test_inputs, test_data_labels, batch_size=4)
test_predict = cls_model.predict(test_inputs)

print("TEST Loss : {:.6f}".format(test_loss))
print("TEST ACC : {:.6f}".format(test_acc))
print("TEST F1-macro : {:.6f}".format(test_macro))
print("TEST F1-micro : {:.6f}".format(test_micro))
print("TEST AUC-macro : {:.6f}".format(roc_auc_score(test_data_labels, test_predict, average='macro')))
print("TEST AUC-micro : {:.6f}".format(roc_auc_score(test_data_labels, test_predict, average='micro')))

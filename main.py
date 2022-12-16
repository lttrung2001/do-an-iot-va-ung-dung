# Note: Tìm hiểu feature extraction using RBM, LSTM units, RBM transform

import numpy as np
import pandas as pd

from sklearn.preprocessing import minmax_scale, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

# Đọc train, test
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_valid = pd.read_csv('valid.csv')

# Loại bỏ khoảng trắng thừa tiêu đề cột
df.columns = df.columns.str.replace(' ', '')
df_test.columns = df.columns.str.replace(' ', '')
df_valid.columns = df.columns.str.replace(' ', '')
# Loại bỏ các cột không cần thiết
df.drop(columns=['Unnamed:0', 'Unnamed:0.1', 'FlowID', 'SourceIP', 'DestinationIP', 'Timestamp', 'SimillarHTTP', 'SourcePort', 'DestinationPort'], axis=1, inplace=True)
df_test.drop(columns=['Unnamed:0', 'Unnamed:0.1', 'FlowID', 'SourceIP', 'DestinationIP', 'Timestamp', 'SimillarHTTP', 'SourcePort', 'DestinationPort'], axis=1, inplace=True)
df_valid.drop(columns=['Unnamed:0', 'Unnamed:0.1', 'FlowID', 'SourceIP', 'DestinationIP', 'Timestamp', 'SimillarHTTP', 'SourcePort', 'DestinationPort'], axis=1, inplace=True)
cols = list(df.columns)
cols.remove('Label')
# Làm sạch dữ liệu
import sys
df.replace([np.inf, ], sys.float_info.max, inplace=True)
df_test.replace([np.inf, ], sys.float_info.max, inplace=True)
df_valid.replace([np.inf, ], sys.float_info.max, inplace=True)
df.replace([np.NaN, ], 0, inplace=True)
df_test.replace([np.NaN, ], 0, inplace=True)
df_valid.replace([np.NaN, ], 0, inplace=True)

# Mã hóa các nhãn
labels = pd.get_dummies(df['Label'])
labels_test = pd.get_dummies(df_test['Label'])
labels_valid = pd.get_dummies(df_test['Label'])

# Loại bỏ cột nhãn
X_train = df.drop(columns=["Label"], axis=1)
X_test = df_test.drop(columns=["Label"], axis=1)
X_valid = df_test.drop(columns=["Label"], axis=1)
# Mã hóa nhãn: BENIGN = 1, còn lại = 0
Y_train = labels.BENIGN
Y_test = labels_test.BENIGN
Y_valid = labels_valid.BENIGN

# Chuẩn hóa dữ liệu
X_train = minmax_scale(X_train, feature_range=(0, 1))  # 0-1 scaling
X_test = minmax_scale(X_test, feature_range=(0, 1))  # 0-1 scaling
X_valid = minmax_scale(X_valid, feature_range=(0, 1))  # 0-1 scaling

# In shape của bộ train, test
print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)
print(Y_train.shape)
print(Y_test.shape)
print(Y_valid.shape)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Train RBM
num_iter = 10
logistic = linear_model.LogisticRegression(solver="newton-cg", tol=1)
# n_components=X_train.shape[1]
# batch_size=X_train.shape[0]//num_iter
rbm = BernoulliRBM(random_state=42, verbose=1, n_components=20, batch_size=X_train.shape[0]//num_iter)
rbm.learning_rate = 0.01
rbm.n_iter = num_iter
rbm.fit(X_train)
X_train = rbm.transform(X_train)
X_test = rbm.transform(X_test)
X_valid = rbm.transform(X_valid)

# def reconstruct_data(X):
#     xx = X.copy()
#     for _ in range(10):
#         for n in range(xx.shape[0]):
#             xx[n] = rbm.gibbs(xx[n])
#     return xx
# print('Reconstruct train')
# X_train = reconstruct_data(X_train)
# print('Reconstruct test')
# X_test = reconstruct_data(X_test)
# print('Reconstruct valid')
# X_valid = reconstruct_data(X_valid)

# indexes = train[0]
# selected_features_index = [i for i in range(len(indexes)) if indexes[i] == True]
# get_selected_features_name = lambda x: df.columns[x]
# print('Feature importance by RBM:', get_selected_features_name(selected_features_index))

# # Lấy những đặc trưng được chọn
# X_train, X_test, X_valid = X_train[:, selected_features_index], X_test[:, selected_features_index], X_valid[:, selected_features_index]

# Đang test
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dropout, Dense, LSTM, BatchNormalization
from keras.callbacks import EarlyStopping

# Reshape
data_dim = X_train.shape[1]
num_classes = 2
X_train = X_train.reshape(-1, 1, data_dim)
X_test  = X_test.reshape(-1, 1, data_dim)
X_valid  = X_valid.reshape(-1, 1, data_dim)
Y_train = Y_train.to_numpy().reshape(-1, 1, 1)
Y_test = Y_test.to_numpy().reshape(-1, 1, 1)
Y_valid = Y_valid.to_numpy().reshape(-1, 1, 1)

# Build model
def RNN():
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, data_dim)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    return model

num_epochs = 120 
opt = Adam(learning_rate=0.001) # 0.0002
model = RNN()
model.summary()
model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
es = EarlyStopping(patience=20, monitor='val_accuracy', restore_best_weights=True)
# batch_size=X_train.shape[0]//num_epochs
history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=num_epochs, batch_size=X_train.shape[0]//num_epochs, verbose=1, callbacks=[es], shuffle=True)

from matplotlib import pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()

train_evaluation = model.evaluate(X_train, Y_train, return_dict=True)
test_evaluation = model.evaluate(X_test, Y_test, return_dict=True)

for name, value in train_evaluation.items():
    print(f"Train {name}: {value:.4f}")
for name, value in test_evaluation.items():
    print(f"Test {name}: {value:.4f}")

Y_pred = model.predict(X_test).argmax(axis=-1)
Y_test = list(map(lambda y: y[0][0], Y_test))
print(set(Y_test))
print(set(Y_pred))

from sklearn.metrics import confusion_matrix
mtx = confusion_matrix(Y_test, Y_pred)
print(mtx)
from sklearn import metrics
print(
    "Report:\n%s\n"
    % (metrics.classification_report(Y_test, Y_pred))
)
(tp, fp), (fn, tn)  = mtx

accuracy = (tp+tn)/(tp+fp+tn+fn)
precision = (tp)/(tp+fp)
recall = (tp)/(tp+fn)
npv = (tn)/(tn+fn)
miss_rate = (fn)/(tp+fn)
fall_out = (fp)/(tn+fp)
fdr = (fp)/(fp+tp)
false_omission_rate = (fn)/(fn+tn)
f1_score = (2*precision*recall)/(precision+recall)

print(f'[+] Accuracy: {accuracy:.4f}')
print(f'[+] Recall: {recall:.4f}')
print(f'[+] Precision: {precision:.4f}')
print(f'[+] NegativePredictive Value: {npv:.4f}')
print(f'[+] Miss rate: {miss_rate:.4f}')
print(f'[+] Fall-Out: {fall_out:.4f}')
print(f'[+] False Discovery Rate: {fdr:.4f}')
print(f'[+] False Omission Rate: {false_omission_rate:.4f}')
print(f'[+] F1 score: {f1_score:.4f}')

from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
# model.save('result/model_{}.h5'.format(dt_string))


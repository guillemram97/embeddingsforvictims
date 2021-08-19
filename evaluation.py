import numpy as np
import math
import time
import argparse
import os
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from keras.layers import CuDNNLSTM
from tensorflow.compat.v1.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.decomposition import PCA
from tqdm import tqdm
import tensorflow_addons as tfa 
from keras import metrics
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K
#print(tf.config.list_physical_devices('GPU'))
#print('ara se ve')
#print(torch.cuda.is_available())

def plot_metric(history, metric, name):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.savefig(name+metric)

class_weight = {0: 9.1, 1: 2.78, 2: 1.92} #{0: 35., 1: 7.0, 2: 6.0}

f1 = tfa.metrics.F1Score(3)
callback = EarlyStopping(monitor='val_accuracy', patience=5)

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--batch_size', metavar='', type=int,
                    help='batch_size')
parser.add_argument('--n_epochs', metavar='', type=int,
                    help='number of epochs')
parser.add_argument('--model', metavar='', type=str,
                    help='Model that we are using')
parser.add_argument('--size', metavar='', type=int,
                    help="Number of samples that we won't use")
parser.add_argument('--dir', metavar='', type=str,
                    help="Directory for the matrix")
parser.add_argument('--dims', metavar='', type=int,
                    help="Dimensions of the word embeddings")

parser.add_argument('--seed', metavar='', type=int,
                    help="Random seed")

params = parser.parse_args()

os.environ['PYTHONHASHSEED']=str(params.seed)
random.seed(params.seed)
np.random.seed(params.seed)
tf.random.set_seed(params.seed)

print('BATCH SIZE:')
print(params.batch_size)

print('Model')
print(params.model)
print('Directory')
print(params.dir)

XX=np.load('bert/X.npy', mmap_mode='r')
Y=np.load('bert/Y.npy')

X=XX[200000:, :].copy()
Y=Y[200000:]

if params.model[:-1] in ['LR', 'procrustes', 'average', 'context_weight', 'context_weigh1']:
	if params.model[:-1]=='average': A=np.identity(params.dims)
	if params.model[:-1]=='LR': A=np.load(params.dir+'/A_LR'+str(params.size)+'.npy')
	if params.model[:-1]=='procrustes': A=np.load(params.dir+'/A_pro'+str(params.size)+'.npy')
	if params.model[:-1]=='context_weight': A=np.load(params.dir+'/A_co'+str(params.size)+'.npy')
	if params.model[:-1]=='context_weigh1': A=np.load(params.dir+'/A_co'+str(params.size)+'.npy')
	if params.model[:-2]!='context_weigh':
		for idx in range(len(Y)):
			contador=0
			aux=np.zeros((1, params.dims))
			for idx_2 in range(11):
				if (X[idx, idx_2*params.dims:(idx_2+1)*params.dims]**2).sum()!=0.0: ++contador
				aux+=X[idx, idx_2*params.dims:(idx_2+1)*params.dims]
			contador=1
			X[idx, 5*params.dims:6*params.dims]=np.matmul(aux, A)/contador
	else: 
		for idx in range(len(Y)):
			if params.model[:-1]=='context_weigh1': X[idx, 5*params.dims:6*params.dims]=(X[idx, 5*params.dims:6*params.dims]+np.matmul(X[idx, :11*params.dims], A))/2
			else: X[idx, 5*params.dims:6*params.dims]=np.matmul(X[idx, :11*params.dims], A)

if params.model[-1]=='1':
        X=X[:, 5*params.dims:6*params.dims]


print('Final shapes')
print(X.shape)
print(Y.shape)

trainX= X[:int(len(Y)*0.9), :] 
trainY= Y[:int(len(Y)*0.9), :]    
testX= X[int(len(Y)*0.9):, :] 
testY= Y[int(len(Y)*0.9):, :]
np.save('testY', testY)
np.save('testX', testX)

look_back=1
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(1, input_shape=(1, X.shape[1])))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])
history=model.fit(trainX, trainY, epochs=params.n_epochs, shuffle=True, class_weight=class_weight, batch_size=params.batch_size,  validation_split=0.1111, verbose=1, callbacks=[callback])
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

plot_metric(history, 'accuracy', 'figures/'+str(params.batch_size))
plot_metric(history, 'loss', 'figures/'+str(params.batch_size))
plot_metric(history, 'f1_score', 'figures/'+str(params.batch_size))

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

trainScore = f1(trainY, trainPredict).numpy()
print('F1: Train Score: %.2f I - %.2f K - %.2f Other' % (trainScore[0], trainScore[1], trainScore[2]))
testScore = f1(testY, testPredict).numpy()
print('F1: Test Score: %.2f I - %.2f K - %.2f Other' % (testScore[0], testScore[1], testScore[2]))

trainScore=metrics.categorical_accuracy(trainY, trainPredict)
print('Train Score: %.2f Categorical Accuracy' % (trainScore.numpy().mean()))
testScore=metrics.categorical_accuracy(testY, testPredict)
print('Test Score: %.2f Categorical Accuracy' % (testScore.numpy().mean()))

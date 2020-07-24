#!/usr/bin/python

import sys

import time
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,LSTM,Conv1D,Input,concatenate
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from processing import build_dataset
from utils import init_session
from processing import dataTest
from keras.layers.convolutional import ZeroPadding1D

batch_size=int(sys.argv[1])
epochs_num=int(sys.argv[2])
process_datas_dir="file/process_datas.pickle"
log_dir="log/CNN_LSTM4.log"
model_dirs=["file/CNN_LSTM4_1_model","file/CNN_LSTM4_2_model","file/CNN_LSTM4_3_model",
            "file/CNN_LSTM4_4_model","file/CNN_LSTM4_5_model","file/CNN_LSTM4_6_model",
            "file/CNN_LSTM4_7_model","file/CNN_LSTM4_8_model","file/CNN_LSTM4_9_model",
            "file/CNN_LSTM4_10_model"]
def train(train_generator,train_size,input_num,dims_num,model_dir):
    start=time.time()
    inputs=Input(shape=(input_num,dims_num))

    if int(sys.argv[3]) == 0:
        con = inputs
    elif int(sys.argv[3]) == 1:
        con=Conv1D(128,2,activation="relu")(inputs)
    elif int(sys.argv[3]) == 2:
        pathway2 = Conv1D(128, 4,activation="relu")(inputs)
        con=ZeroPadding1D(padding=1)(pathway2)
    elif int(sys.argv[3]) == 3:
        pathway3 = Conv1D(128, 6,activation="relu")(inputs)
        con=ZeroPadding1D(padding=2)(pathway3)
    else:
        pathway1=Conv1D(128,2,activation="relu")(inputs)
        pathway2 = Conv1D(128, 4,activation="relu")(inputs)
        pathway2=ZeroPadding1D(padding=1)(pathway2)
        pathway3 = Conv1D(128, 6,activation="relu")(inputs)
        pathway3=ZeroPadding1D(padding=2)(pathway3)
        con = concatenate([pathway1,pathway2,pathway3])

    #============================
    layer1=LSTM(128)(con)
    layer1=Dropout(0.5)(layer1)
    output=Dense(2,activation="softmax",name="Output")(layer1)
    optimizer=Adam()
    # model=Sequential()
    # model.add(inputs)
    # model.add(con)
    # model.add(layer1)
    #
    # model.add(Dropout(0.5))
    #
    # model.add(MaxPool1D(pool_size=2))
    # model.add(output)
    # call=TensorBoard(log_dir=log_dir,write_grads=True,histogram_freq=0)#将histogram_freq的值由1改成0
    model=Model(inputs=inputs,outputs=output)
    model.compile(optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
    model.fit_generator(train_generator,steps_per_epoch=train_size//(batch_size*epochs_num),epochs=epochs_num,callbacks=None)
#    model.fit_generator(train_generator, steps_per_epoch=5, epochs=5, callbacks=[call])
    model.save(model_dir)
    end=time.time()
    print("Training took %f s"%(end-start))
    print(model.summary())

if __name__=="__main__":
    train_generators, test_generators, train_size, test_size, input_num, dims_num= build_dataset(
        batch_size)
    train(train_generators[0], train_size, input_num, dims_num,model_dir=model_dirs[0])
    dataTest(model_dirs[0], test_generators[0], test_size, input_num, dims_num, batch_size)

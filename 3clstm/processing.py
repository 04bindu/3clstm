#!/usr/bin/python

import sys

from utils import GeneSeg
import csv,pickle,random,json
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split,StratifiedKFold,KFold
import tensorflow as tf

from keras.models import load_model
from keras import backend as K
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as ac

vec_dir="file/word2vec.pickle"
pre_datas_trains=["file/pre_datas_train1.csv","file/pre_datas_train2.csv","file/pre_datas_train3.csv",
                 "file/pre_datas_train4.csv","file/pre_datas_train5.csv","file/pre_datas_train6.csv",
                 "file/pre_datas_train7.csv","file/pre_datas_train8.csv","file/pre_datas_train9.csv",
                 "file/pre_datas_train10.csv"]
pre_datas_tests=["file/pre_datas_test1.csv","file/pre_datas_test2.csv","file/pre_datas_test3.csv",
                "file/pre_datas_test4.csv","file/pre_datas_test5.csv","file/pre_datas_test6.csv",
                "file/pre_datas_test7.csv","file/pre_datas_test8.csv","file/pre_datas_test9.csv",
                "file/pre_datas_test10.csv"]
# pre_datas_validation="file/pre_datas_validation.csv"
# process_datas_dir="file/process_datas.pickle"

def pre_process():
    with open(vec_dir,"rb") as f :
        word2vec=pickle.load(f)
        dictionary=word2vec["dictionary"]
        reverse_dictionary=word2vec["reverse_dictionary"]
        embeddings=word2vec["embeddings"]
    xssed_data=[]
    normal_data=[]
    with open("data/xssed.csv","r",encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=["payload"])
        for row in reader:
            payload=row["payload"]
            word=GeneSeg(payload)
            xssed_data.append(word)
    with open("data/normal_examples.csv","r",encoding="utf-8") as f:
        reader=csv.reader(f)
        reader = csv.DictReader(f, fieldnames=["payload"])
        for row in reader:
            payload=row["payload"]
            word=GeneSeg(payload)
            normal_data.append(word)
    xssed_num=len(xssed_data)
    normal_num=len(normal_data)
    xssed_labels=[1]*xssed_num
    normal_labels=[0]*normal_num
    datas=xssed_data+normal_data
    labels=xssed_labels+normal_labels
    labels=to_categorical(labels)
    def to_index(data):
        d_index=[]
        for word in data:
            if word in dictionary.keys():
                d_index.append(dictionary[word])
            else:
                d_index.append(dictionary["UNK"])
        return d_index
    datas_index=[to_index(data) for data in datas]
    datas_index=pad_sequences(datas_index,value=-1)
    rand=random.sample(range(len(datas_index)),len(datas_index))
    datas=[datas_index[index] for index in rand]
    labels=[labels[index] for index in rand]


    if sys.argv[1] == 'kfold':
        folder=KFold(n_splits=10,shuffle=False,random_state=0)
        number=0
        for train,test in folder.split(datas,labels):

            train = np.random.choice(train, size = len(train)/int(sys.argv[3]), replace = False)
            test = np.random.choice(test, size = len(test)/int(sys.argv[3]), replace = False)

            train_datas=[datas[i] for i in train]
            test_datas=[datas[i] for i in test]
            train_labels=[labels[i] for i in train]
            test_labels=[labels[i] for i in test]
            train_size=len(train_labels)
            test_size=len(test_labels)


            input_num=len(train_datas[0])
            dims_num = embeddings["UNK"].shape[0]
            word2vec["train_size"]=train_size
            word2vec["test_size"]=test_size
            word2vec["input_num"]=input_num
            word2vec["dims_num"]=dims_num

            with open(pre_datas_trains[number],"w") as f:
                for i in range(train_size):
                    data_line=str(train_datas[i].tolist())+"|"+str(train_labels[i].tolist())+"\n"
                    f.write(data_line)

            with open(pre_datas_tests[number],"w") as f:
                for i in range(test_size):
                    data_line=str(test_datas[i].tolist())+"|"+str(test_labels[i].tolist())+"\n"
                    f.write(data_line)

            number=number+1

    else:
        train_datas, test_datas, train_labels, test_labels = train_test_split(datas, labels, test_size = float(sys.argv[2]))

        train_size=len(train_labels)
        test_size=len(test_labels)

        input_num=len(train_datas[0])
        dims_num = embeddings["UNK"].shape[0]
        word2vec["train_size"]=train_size
        word2vec["test_size"]=test_size
        word2vec["input_num"]=input_num
        word2vec["dims_num"]=dims_num

        with open(pre_datas_trains[0],"w") as f:
            for i in range(train_size):
                data_line=str(train_datas[i].tolist())+"|"+str(train_labels[i].tolist())+"\n"
                f.write(data_line)

        with open(pre_datas_tests[0],"w") as f:
            for i in range(test_size):
                data_line=str(test_datas[i].tolist())+"|"+str(test_labels[i].tolist())+"\n"
                f.write(data_line)


    with open(vec_dir, "wb") as f:
        pickle.dump(word2vec, f)

def data_generator(data_dir):
    #value = tf.data.TextLineDataset([data_dir])
    #reader = tf.TextLineReader()
    #queue = tf.train.string_input_producer([data_dir])
    #_, value = reader.read(queue)
    #coord = tf.train.Coordinator()
    #sess = tf.Session()
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #while True:
    #    v = value
    #    [data, label] = v.split(b"|")
    #    data = np.array(json.loads(data.decode("utf-8")))
    #    label = np.array(json.loads(label.decode("utf-8")))
    #    yield (data, label)
    #coord.request_stop()
    #coord.join(threads)
    #sess.close()
    df = tf.data.TextLineDataset([data_dir])
    #print(df)
    for line in df:
        [data, label] = tf.strings.split(line, b"|").numpy()
        data = np.array(json.loads(data.decode("utf-8")))
        label = np.array(json.loads(label.decode("utf-8")))
        yield (data, label)

def batch_generator(datas_dir,datas_size,batch_size,embeddings,reverse_dictionary,train=True):
    batch_data = []
    batch_label = []
    generator=data_generator(datas_dir)
    n=0
    while True:
        for i in range(batch_size):
            data,label=next(generator)
            data_embed = []
            for d in data:
                if d != -1:
                    data_embed.append(embeddings[reverse_dictionary[d]])
                else:
                    data_embed.append([0.0] * len(embeddings["UNK"]))
            batch_data.append(data_embed)
            batch_label.append(label)
            n+=1
            if not train and n==datas_size:
                break
        if not train and n == datas_size:
            yield (np.array(batch_data), np.array(batch_label))
            break
        else:
            yield (np.array(batch_data),np.array(batch_label))
            batch_data = []
            batch_label = []
def build_dataset(batch_size):
    with open(vec_dir, "rb") as f:
        word2vec = pickle.load(f)
    embeddings = word2vec["embeddings"]
    reverse_dictionary = word2vec["reverse_dictionary"]
    train_size=word2vec["train_size"]
    test_size=word2vec["test_size"]
    dims_num = word2vec["dims_num"]
    input_num =word2vec["input_num"]

    train_generators=[]
    test_generators=[]

    for i in range(10):
        train_generator = batch_generator(pre_datas_trains[i],train_size,batch_size,embeddings,reverse_dictionary)
        test_generator = batch_generator(pre_datas_tests[i],test_size,batch_size,embeddings,reverse_dictionary,train=False)
        train_generators.append(train_generator)
        test_generators.append(test_generator)

    return train_generators,test_generators,train_size,test_size,input_num,dims_num


# AUC for a binary classifier
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P


def dataTest(model_dir,test_generator,test_size,input_num,dims_num,batch_size):
    model=load_model(model_dir,custom_objects={'auc':auc,'binary_PFA':binary_PFA,'binary_PTA':binary_PTA})
    labels_pre=[]
    labels_true=[]
    batch_num=test_size//batch_size+1
    steps=0
    for batch,labels in test_generator:
        if len(labels)==batch_size:
            labels_pre.extend(model.predict_on_batch(batch))
        else:
            batch=np.concatenate((batch,np.zeros((batch_size-len(labels),input_num,dims_num))))
            labels_pre.extend(model.predict_on_batch(batch)[0:len(labels)])
        labels_true.extend(labels)
        steps+=1
        print("%d/%d batch"%(steps,batch_num))
    labels_pre=np.array(labels_pre).round()
    def to_y(labels):
        y=[]
        for i in range(len(labels)):
            if labels[i][0]==1:
                y.append(0)
            else:
                y.append(1)
        return y
    y_true=to_y(labels_true)
    y_pre=to_y(labels_pre)
    precision=precision_score(y_true,y_pre)
    recall=recall_score(y_true,y_pre)
    accuracy=accuracy_score(y_true,y_pre)
    f1=f1_score(y_true,y_pre)

    print("Precision score is :",precision)
    print("Recall score is :",recall)
    print("Accuracy score is : ",accuracy)
    print("F1 score is : ",f1)

    fpr, tpr, thresholds = roc_curve(y_true, y_pre)
    roc_auc = ac(fpr, tpr)
    print("FP rate is :",fpr)
    print("TP rate is :",tpr)
    print("Roc_AOC is :",roc_auc)

if __name__=="__main__":
    pre_process()

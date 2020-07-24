import csv,pickle,time
from collections import Counter
from sklearn.manifold import TSNE
from utils import GeneSeg
from gensim.models.word2vec import Word2Vec
learning_rate=0.001
vocabulary_size=3000
batch_size=128
embedding_size=128
num_skips=4
skip_window=5
num_sampled=64
num_iter=5
log_dir="word2vec.log"
vec_dir="file/word2vec.pickle"

start=time.time()
words=[]
datas=[]
with open("data/xssed.csv","r",encoding="utf-8") as f:
    reader=csv.DictReader(f,fieldnames=["payload"])
    for row in reader:
        payload=row["payload"]
        word=GeneSeg(payload)
        datas.append(word)
        words+=word

#构建数据集
def build_dataset(datas,words):
    count=[["UNK",-1]]
    counter=Counter(words)
    count.extend(counter.most_common(vocabulary_size-1))
    vocabulary=[c[0] for c in count]
    data_set=[]
    for data in datas:
        d_set=[]
        for word in data:
            if word in vocabulary:
                d_set.append(word)
            else:
                d_set.append("UNK")
                count[0][1]+=1
        data_set.append(d_set)
    return data_set
data_set=build_dataset(datas,words)

model=Word2Vec(data_set,size=embedding_size,window=skip_window,negative=num_sampled,iter=num_iter,sg=1)
embeddings=model.wv

def save(embeddings):
    dictionary=dict([(embeddings.index2word[i],i)for i in range(len(embeddings.index2word))])
    reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    word2vec={"dictionary":dictionary,"embeddings":embeddings,"reverse_dictionary":reverse_dictionary}
    with open(vec_dir,"wb") as f:
        pickle.dump(word2vec,f)
save(embeddings)
end=time.time()
print("Job took ",end-start)

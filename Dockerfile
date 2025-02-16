FROM tensorflow/tensorflow

# Install dependencies
RUN pip install \
    scikit-learn \
    nltk \
    keras \
    gensim

RUN mkdir /3clstm/
RUN mkdir /results/
COPY 3clstm/ /3clstm/
RUN mkdir /3clstm/file

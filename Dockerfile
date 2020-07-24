FROM tensorflow/tensorflow

# Install dependencies
RUN pip install \
    scikit-learn \
    nltk \
    keras \
    gensim

COPY 3clstm/ /
RUN mkdir 3clstm/3clstm/file

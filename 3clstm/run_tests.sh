
python word2vec_gensim.py
python processing.py 'kfold' 0 1

# Control
python CNN_LSTM4.py 100 1 4 > "../results/control.log"

# Batch sizes
python CNN_LSTM4.py 200 1 4 > "../results/batch_200.log"
python CNN_LSTM4.py 50 1 4 > "../results/batch_50.log"
python CNN_LSTM4.py 500 1 4 > "../results/batch_500.log"
python CNN_LSTM4.py 100 2 4 > "../results/epochs_2.log"

# CNN Pathways
python CNN_LSTM4.py 100 1 0 > "../results/cnn_none.log"
python CNN_LSTM4.py 100 1 1 > "../results/cnn_only_1.log"
python CNN_LSTM4.py 100 1 2 > "../results/cnn_only_2.log"
python CNN_LSTM4.py 100 1 3 > "../results/cnn_only_3.log"

python processing.py 'tts' 0.33 1

# Train-test split
python CNN_LSTM4.py 100 1 4 > "../results/split_2_1.log"

python processing.py 'tts' 0.5 1

# Train-test split
python CNN_LSTM4.py 100 1 4 > "../results/split_1_1.log"

python processing.py 'kfold' 0 2

# Row sampling
python CNN_LSTM4.py 100 1 4 > "../results/sampling_50.log"

python processing.py 'kfold' 0 4

# Row sampling
python CNN_LSTM4.py 100 1 4 > "../results/sampling_25.log"

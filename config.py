MATCH = 1
NO_MATCH = 0
MAX_DIST = 2000
lm_hidden = 768
c_em = 256
n_em = 256
a_em = 256
max_seq_len = 128
device = "cpu"
dropout = 0.2
epochs = 10
batch_size = 32
lr = 3e-5
path = "./data/train_valid_test/"
n_path = "./data/neighborhood_train_valid_test/"
save_path = "./saved_models/best_model_"

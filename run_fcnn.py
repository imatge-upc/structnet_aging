import os

data_dir = "./data/"
res_dir  = "./results/"

seed = 0
epochs = 100
bs=32

lr = 1e-3
wd = 1e-4
do = 0.1
bn = True

fcnn_num_layers = 2
fcnn_hid_dim    = 32

os.system("python3 train_fcnn.py " +

            "--data_dir {} ".format(data_dir) +
            "--res_dir {} ".format(res_dir) +
                
            "--train_size 0.6 " +
            "--valid_size 0.2 " +
            "--test_size 0.2 " +

            "--seed {} ".format(0) +
            "--epochs {} ".format(epochs) +
            "--batch_size {} ".format(bs) +

            "--lr {} ".format(lr) +
            "--weight_decay {} ".format(wd) +
            "--dropout {} ".format(do) +
            ("--bn {} ".format(True) if bn else "") +

            "--fcnn_num_layers {} ".format(fcnn_num_layers) +
            "--fcnn_hid_dim {} ".format(fcnn_hid_dim)
        )
import os

data_dir = "./data/"
res_dir  = "./results/"

seed = 0
epochs = 10
bs=128

nwk_method = "rlasso"
nwk_alpha  = 5e-2
nwk_max_iter = 1000

lr = 1e-3
wd = 0.0
do = 0.1
bn = True

msgp_mode      = "PE"
gpool_mode     = "RA"
gnn_num_layers = 2
gnn_hid_dim    = 5

os.system("python3 train_gnn.py " +

            "--data_dir {} ".format(data_dir) +
            "--res_dir {} ".format(res_dir) +
                
            "--train_size 0.6 " +
            "--valid_size 0.2 " +
            "--test_size 0.2 " +

            "--seed {} ".format(0) +
            "--epochs {} ".format(epochs) +
            "--batch_size {} ".format(bs) +


            "--nwk_method {} ".format(nwk_method) +
            "--nwk_alpha {} ".format(nwk_alpha) +
            "--nwk_max_iter {} ".format(nwk_max_iter) +

            "--lr {} ".format(lr) +
            "--weight_decay {} ".format(wd) +
            "--dropout {} ".format(do) +
            ("--bn {} ".format(True) if bn else "") +

            "--msgp_mode {} ".format(msgp_mode) +
            "--gpool_mode {} ".format(gpool_mode) +
            "--gnn_num_layers {} ".format(gnn_num_layers) +
            "--gnn_hid_dim {} ".format(gnn_hid_dim)
        )
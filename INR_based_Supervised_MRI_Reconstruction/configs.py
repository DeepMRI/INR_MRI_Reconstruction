import os


class config(object):

    # hardware
    device = 'cpu'                  # device: cpu, cuda:0, cuda:1 and so on
    batch_size = 2

    is_train = True
    is_eval = True

    # path
    snap_path = 'results\\knee\\scale=4-5-6'
    datapath = 'E:\\dataset\\fastMRI_knee\\Reconstruction'
    eval_path = 'E:\\dataset\\fastMRI_knee\\Reconstruction\\restored\\msr'

    # data params
    down_scale = (4, 5, 6)              # MRI undersample scale
    eval_scale = (4, 5, 6)
    keep_center = 0.08
    is_pre_combine = False

    # train params
    n_iters = 100000                 # train iterations
    log_step = 100                   # log file record step
    val_step = 1000                  # validation step
    load_checkpoint = False         # load checkpoint

    # optimizer params
    lr = 1e-3                       # learning rate
    beta1 = 0.9
    beta2 = 0.99
    weight_decay = 0.0
    scheduler_step = 10000

    # network params:
    in_dim = 15                      # coordinates dimension: 2D or 3D
    out_dim = 1                     # pred image
    hidden_dim = 256
    num_layer = 8
    skips = (3, 6)                   # skip connection layer list
    nonlinearity = 'swish'           # activation function if use 'mlp' network: elu, relu, lrelu, swish
    use_encoder = True

    # positional encoding params:
    pos_encoding = True
    include_coord = True            # reserve original coordinates or not
    pos_fre_num = 128                # enconding frequency num
    pos_scale = 1                   # enconding coefficient scale

    def __init__(self, record=True):

        if record:
            self.record()

    def to_dict(self):
        return {a: getattr(self, a)
                for a in sorted(dir(self))
                if not a.startswith("__") and not callable(getattr(self, a))}

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for key, val in self.to_dict().items():
            print(f"{key:30} {val}")
        # for a in dir(self):
        #     if not a.startswith("__") and not callable(getattr(self, a)):
        #         print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def record(self, subpath=None):

        if subpath is not None:
            path = os.path.join(self.snap_path, subpath)
        else:
            path = self.snap_path

        # create folder
        if not os.path.exists(path):
            os.makedirs(path)

        config_file = open(os.path.join(path, 'configs.txt'), 'w')
        for key, val in self.to_dict().items():
            config_file.write(f"{key:30} \t {val}\n")
        config_file.close()

class TrainConfig:
    batch_size = 32
    use_fuse_model = False
    select_model = "all_fuse"
    dataset_select = "plan"
    use_metrics = False
    use_log = False
    model_path = None
    res_path = None
    model_name = None
    lr = 3e-4
    time_t = 1.
    use_softmax = False
    use_margin_loss = False
    use_label_loss = False
    use_weight_loss = False
    use_threshold_loss = False
    margin_loss_type = "MarginLoss"
    epoch = 50
    opt_threshold = 0.05
    train_name = None

class Args:
    bs = 1024
    lr = 0.001
    epochs = 200
    clip_size = 50
    embed_size = 64
    pred_hid = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    dropout = 0.1
    sch_decay = 0.6
    device = 'cuda:0'
    newpath = './results/full/cost/'
    to_predict = 'cost'
    input_emb = 1063
    use_sample = True
    
class ArgsPara:
    diff_weight = 0.05
    share_weight = 0.05
    margin_weight = 1.0
    # margin_weight = 0.05
    mul_label_weight = 1.0
    ts_weight = 1.0
    cons_weigtht = 1.0
    threshold_factor = 1
    std_threshold = 0.05
    pred_type = "pred_opt"
    dataset = "0.05rate"
    

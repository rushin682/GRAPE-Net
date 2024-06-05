import argparse

# Training settings
def train_arguments():
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')

    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--drop_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--jk', type=str, default='sum',
                        help='Jumping knowledge aggregations : last | sum')
    parser.add_argument('--graph_pooling', type=str, default='gmt',
                        help='Graph pooling type : sum | mean | max | attention | set2set')


    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--n_epochs', type=int, default=65,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers (default: 0)')
    parser.add_argument('--b', type=float, default=0.1,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_policy', type=str, default='step',
                        help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=30,
                        help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--l2_weight_decay', type=float, default=0.01,
                        help='The weight decay for L2 Norm in Adam optimizer')

    parser.add_argument('--dataset', type=str, default="tcga",
                        help='dataset name (default: tcga)')
    parser.add_argument('--phase', type=str, default="train",
                        help='dataset phase : train | test | plot')
    parser.add_argument('--n_classes', type=int, default=3,
                        help='Number of classes')
    parser.add_argument('--data_config', type=str, default="ctranspath_files",
                        help='dataset config i.e tile size and bkg content (default: simclr_8Conn_files)')
    parser.add_argument('--fdim', type=int, default=768,
                        help='expected feature dim for each node.')

    parser.add_argument('--n_folds', type=int, default=5,
                        help='total number of folds.')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation.  Should be less then 10.')
    parser.add_argument('--no_val', action='store_true', help='no validation set for tuning')

    parser.add_argument('--config_file', type=str, default="configs/config.yaml",
                        help='parameter and hyperparameter config i.e all values for model and dataset parameters')
    parser.add_argument('--project_name', type=str, default=None,
                        help='parameter and hyperparameter config i.e all values for model and dataset parameters')
    
    return parser.parse_args()
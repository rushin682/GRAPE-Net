import os
from tqdm import tqdm
import argparse
import time
import numpy as np
import pandas as pd
import random
import shutil
from arguments import train_arguments
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, average_precision_score

import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch.optim as optim
import torch.nn.functional as F
import wandb

from lightning.pytorch.loggers import WandbLogger

from graph_perceiver import GraphPerceiver
from dataloaders import GraPeDataset
from util import read_file, separate_data, get_scheduler, find_dataset_using_name, EarlyStopping, BinaryCrossEntropyLoss
from evaluate import Evaluator, plot_confusion_matrix

# multicls_criterion = torch.nn.CrossEntropyLoss()
multicls_criterion = BinaryCrossEntropyLoss()
# os.environ['WANDB_DISABLED'] = 'True'

def train(config, epoch, model, device, loader, optimizer, scheduler, train_evaluator):
    model.train()

    train_losses = []
    y_true = []
    y_pred = []
    y_prob = [] 
    pred = None
    for i, graphs in enumerate(tqdm(loader)):

        step = len(loader) * epoch + i

        graphs = graphs.to(device)

        if graphs.x.shape[0] == 1 or graphs.batch[-1] == 0:
            pass
        else:
            pred = model(graphs)
            optimizer.zero_grad()

            loss = multicls_criterion(pred.to(torch.float32), graphs.y.view(-1,))
            # add flooding here
            loss = (loss-config.b).abs() + config.b
            loss.backward()
            optimizer.step()

            wandb.log({'mini-batch-loss/train': loss})

            train_losses.append(loss.item())

        y_true.append(graphs.y.view(-1,1).detach().cpu())
        y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())
        y_prob.append(F.softmax(pred.detach(), dim=1).cpu())

    scheduler.step()

    avg_loss = torch.mean(torch.tensor(train_losses))
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    y_prob = torch.cat(y_prob, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}

    return train_evaluator.eval(input_dict), avg_loss
    # return avg_loss

def eval(model, device, loader, evaluator):
    model.eval()

    val_losses = []
    y_true = []
    y_pred = []
    y_prob = []
    for step, graphs in enumerate(tqdm(loader, desc="Iteration")):

        graphs = graphs.to(device)

        if graphs.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(graphs)

                loss = multicls_criterion(pred.to(torch.float32), graphs.y.view(-1,))
                val_losses.append(loss.item())

            y_true.append(graphs.y.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())
            y_prob.append(F.softmax(pred.detach(), dim=1).cpu())
            # print(y_prob[-1])

    avg_loss = torch.mean(torch.tensor(val_losses))

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    y_prob = torch.cat(y_prob, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}

    return evaluator.eval(input_dict), avg_loss

def main():
    
    ### User Arguments
    args = train_arguments()
    with open(args.config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    # Logger: Lightning
    wandblogger = WandbLogger(project=config.project_name)
    
    ### Wandb configurations: PyTorch
    """ if args.project_name is None:
        # Add the project name as "Graph-Perciever-{}-{}" where {} is the current month in words and date.
        args.project_name = "Graph-Perciever_{}".format(time.strftime("%B-%d"))
        

    # wandb configurations & creating reqd. folders
    wandb.init(project=args.project_name, config=args.config_file)
    # Add the wandb.run.name as "Graph-Perciever-{}-{}" where {} is the current month in words and date.
    wandb.run.name = "Graph-Perciever_{}".format(time.strftime("%B-%d")) + "_fold_" + str(args.fold_idx)
    wandb.config.update({'fold_idx': args.fold_idx,
                         'run_name': wandb.run.name,
                         'log_path': os.path.join('logs', wandb.run.name),
                         'device': args.device}, allow_val_change=True)

    config = wandb.config
    os.makedirs(config.log_path, exist_ok=True)

    print(config) """

    ### set up seeds and gpu device: PyTorch
    """  random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    
    ### cuda device settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:" + str(config.device)) if torch.cuda.is_available() else torch.device("cpu") 
    """

    # model loading: Lightning
    model = GraphPerceiver(config)

    # dataset loading: Lightning
    data_module = GraPeDataset(config)

    ### automatic dataloading and splitting: PyTorch
    root = os.path.join('/SeaExp/Rushin/datasets', config.dataset.upper(), config.data_config)

    wsi_file = os.path.join('/SeaExp/Rushin/datasets', config.dataset.upper(), '%s_%s.txt' % (config.dataset.upper(), config.phase))
    wsi_ids = read_file(wsi_file)

    train_val_ids, test_ids, train_val_labels = separate_data(wsi_ids, config.seed, config.n_folds, config.fold_idx)

    dataset_class = find_dataset_using_name(config.dataset)
    isTrain = True if config.phase == 'train' else False

    ########################################################################################

    if config.no_val:
        train_dataset = dataset_class(root, train_val_ids, config.fdim, config.n_classes, isTrain=isTrain, transform=T.ToSparseTensor(remove_edge_index=False))
        np.savetxt(os.path.join(config.log_path, f'{config.run_name}_fold_{config.fold_idx}_train.txt'), train_val_ids, fmt='%s')
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)

    else:
        print("Use Train, Val, Test CV")
        train_ids, valid_ids = train_test_split(train_val_ids, stratify=train_val_labels, random_state=config.seed, test_size=0.25)
        
        train_dataset = dataset_class(root, train_ids, config.fdim, config.n_classes, isTrain=isTrain, transform=T.ToSparseTensor(remove_edge_index=False))
        np.savetxt(os.path.join(config.log_path, f'{config.run_name}_fold_{config.fold_idx}_train.txt'), train_ids, fmt='%s')
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)

        valid_dataset = dataset_class(root, valid_ids, config.fdim, config.n_classes, isTrain=isTrain, transform=T.ToSparseTensor(remove_edge_index=False))
        np.savetxt(os.path.join(config.log_path, f'{config.run_name}_fold_{config.fold_idx}_val.txt'), valid_ids, fmt='%s')
        valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    
    test_dataset = dataset_class(root, test_ids, config.fdim, config.n_classes, isTrain=isTrain, transform=T.ToSparseTensor(remove_edge_index=False))
    np.savetxt(os.path.join(config.log_path, f'{config.run_name}_fold_{config.fold_idx}_test.txt'), test_ids, fmt='%s')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)

    # evaluation objects: PyTorch
    train_evaluator = Evaluator(train_dataset)
    valid_evaluator = Evaluator(valid_dataset) 
    test_evaluator = Evaluator(test_dataset)
 
    

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-08, 
                            weight_decay=config.l2_weight_decay, 
                            amsgrad=False)

    scheduler = get_scheduler(optimizer, config)

    # initialize the early_stopping object: PyTorch
    early_stopping = EarlyStopping(patience=5, delta=0.02, verbose=True, path=os.path.join(config.log_path, 'checkpoint.pth'))

    if isTrain:
        val_auc_log, val_recall_log = 0, 0
        best_loss_epoch, best_recall_epoch, best_auc_epoch = 0,0,0
        val_loss_log = float('inf')

        wandb.watch(model)

        for epoch in range(config.n_epochs+1):
            print("=====Epoch {}".format(epoch))
            print("Train Loader length", len(train_loader))
            
            # logs loss per iteration and returns avg loss per epoch
            train_perf, train_loss = train(config, epoch, model, device, train_loader, optimizer, scheduler, train_evaluator)

            if not config.no_val:
                print('Evaluating...')
                valid_perf, valid_loss = eval(model, device, valid_loader, valid_evaluator)

                if epoch > 10:
                    if np.mean(valid_perf['rocauc']) > val_auc_log:
                        val_auc_log = np.mean(valid_perf['rocauc'])
                        best_auc_epoch = epoch
                        best_model_save_path = os.path.join(config.log_path, f'best_rocauc_model_{config.run_name}.pth')
                        torch.save(model.state_dict(), best_model_save_path)
                    if np.mean(valid_perf['recall']) > val_recall_log:
                        val_recall_log = np.mean(valid_perf['recall'])
                        best_recall_epoch = epoch
                        best_model_save_path = os.path.join(config.log_path, f'best_recall_model_{config.run_name}.pth')
                        torch.save(model.state_dict(), best_model_save_path)
                    if valid_loss < val_loss_log:
                        val_loss_log = valid_loss
                        best_loss_epoch = epoch
                        best_model_save_path = os.path.join(config.log_path, f'best_loss_model_{config.run_name}.pth')
                        torch.save(model.state_dict(), best_model_save_path)                  

                # save model named by epoch every 10 epochs
                if epoch % 10 == 0:
                    model_save_path = os.path.join(config.log_path, f'epoch_{epoch}_model_{config.run_name}.pth')
                    torch.save(model.state_dict(), model_save_path)

            # print('Train', train_perf)
            print('Validation', valid_perf)

            metrics = {'loss/train': train_loss,
                    'rocauc/train': train_perf['rocauc'],
                    'recall/train': np.mean(train_perf['recall']),
                    'loss/val': valid_loss,                   
                    'rocauc/valid': valid_perf['rocauc'],
                    'recall/valid': np.mean(valid_perf['recall']),
                    'epoch': epoch,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    }

            wandb.log(metrics)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    wandb.log({f'gradient/{name}': wandb.Histogram(param.grad.data.cpu().numpy())})

            # write a snippet for early stopping when the val_loss doesn't change with a delta of 0.02 for 5 epochs
            if epoch > 30:
                early_stopping(valid_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        if config.no_val:
            torch.save(model.state_dict(), os.path.join(config.log_path, f'best_model_{config.run_name}.pth'))
        else:
            print('Saving Final Model')
            final_model_save_path = os.path.join(config.log_path, f'final_model_{config.run_name}.pth')
            torch.save(model.state_dict(), final_model_save_path)

            print(f'Fold {config.fold_idx} - best epoch: {best_auc_epoch} with Val AUROC: {val_auc_log}')
            shutil.copy2(os.path.join(config.log_path, f'best_rocauc_model_{config.run_name}.pth'), os.path.join(config.log_path, f'best_rocauc_model_{config.run_name}_epoch{best_auc_epoch}.pth'))

            print(f'Fold {config.fold_idx} - best epoch: {best_recall_epoch} with Val Recall: {val_recall_log}')
            shutil.copy2(os.path.join(config.log_path, f'best_recall_model_{config.run_name}.pth'), os.path.join(config.log_path, f'best_recall_model_{config.run_name}_epoch{best_recall_epoch}.pth'))

            print(f'Fold {config.fold_idx} - best epoch: {best_loss_epoch} with Val Loss: {val_loss_log}')
            shutil.copy2(os.path.join(config.log_path, f'best_loss_model_{config.run_name}.pth'), os.path.join(config.log_path, f'best_loss_model_{config.run_name}_epoch{best_loss_epoch}.pth'))
        print('Final epoch validation score: {}'.format(valid_perf))

    valid_curve = []
    train_curve = []
    test_curve = []
    valid_cm_plots = []
    test_cm_plots = []
    
    print('Finished Training.....start testing')
    testing_metrics = ['rocauc', 'recall', 'loss']

    for metric in testing_metrics:
        # test model on test set (in-domain)
        best_model_load_path = os.path.join(config.log_path, f'best_{metric}_model_{config.run_name}.pth')
        
        model.load_state_dict(torch.load(best_model_load_path))
        model = model.to(device)
        
        test_perf, test_loss = eval(model, device, test_loader, test_evaluator)
        print('Final evaluation with best {} - Test scores: {}'.format(metric, test_perf))

        final_cm_plot = plot_confusion_matrix(test_perf['cm'], list(test_dataset.classdict.keys()), title='fold{}(Test accuracy={:0.2f})'.format(config.fold_idx+1, np.mean(test_perf['acc'])))
        wandb.log({"{}/ConfusionMatrix".format(metric): final_cm_plot})

        with open(os.path.join(config.log_path, config.run_name+'_best_{}_final_test_perf.txt'.format(metric)), 'w') as f:
            for key, value in test_perf.items():
                f.write('%s:%s\n' % (key, value))

    wandb.finish()
if __name__ == "__main__":
    main()

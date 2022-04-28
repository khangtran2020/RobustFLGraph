import sys, os

sys.path.append(os.path.abspath('..'))

import time
import pickle
import numpy as np
import copy

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from utils.datareader import GraphData, DataReader
from utils.batch import collate_batch
from model.gcn import GCN
from model.sage import GraphSAGE
from config import parse_args


def run(args):
    assert torch.cuda.is_available(), 'no GPU available'
    cpu = torch.device('cpu')
    cuda = torch.device('cuda')

    # load data into DataReader object
    dr = DataReader(args)

    loaders = {}
    for i in args.num_client:
        loaders['client_{}'.format(i)] = {}
        for split in ['train', 'test']:
            if split == 'train':
                gids = dr.data['client_{}'.format(i)]['train']
            else:
                gids = dr.data['client_{}'.format(i)]['test']
            gdata = GraphData(dr, gids)
            loader = DataLoader(gdata,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=collate_batch)
            # data in loaders['train/test'] is saved as returned format of collate_batch()
            loaders['client_{}'.format(i)][split] = loader
        print('train %d, test %d' % (
            len(loaders['client_{}'.format(i)]['train'].dataset), len(loaders['client_{}'.format(i)]['test'].dataset)))

    # prepare model
    in_dim = loaders['client_{}'.format(0)]['train'].dataset.num_features
    out_dim = loaders['client_{}'.format(0)]['train'].dataset.num_classes
    models = {}
    if args.model == 'gcn':
        global_model = GCN(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
        for i in range(args.num_client):
            models['client_{}'.format(i)] = GCN(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    elif args.model == 'sage':
        global_model = GraphSAGE(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
        for i in range(args.num_client):
            models['client_{}'.format(i)] = GraphSAGE(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    else:
        raise NotImplementedError(args.model)

    # print('\nInitialize model')
    # print(model)
    global_weights = global_model.state_dict()
    train_params = {}
    for i in range(args.num_client):
        train_params['client_{}'.format(i)] = list(
            filter(lambda p: p.requires_grad, models['client_{}'.format(i)].parameters()))
    # print('N trainable parameters:', np.sum([p.numel() for p in train_params]))

    predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()
    loss_fn = F.cross_entropy
    # training
    global_model.to(cuda)
    global_model.train()
    for e in range(args.fl_epochs):
        local_weights = []
        for i in range(args.num_client):
            model = models['client_{}'.format(i)]
            global_model.load_state_dict(global_weights)
            optimizer = optim.Adam(train_params['client_{}'.format(i)], lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))
            scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)
            model.to(cuda)
            for epoch in range(args.train_epochs):
                model.train()
                start = time.time()
                train_loss, n_samples = 0, 0
                for batch_id, data in enumerate(loaders['client_{}'.format(i)]['train']):
                    for j in range(len(data)):
                        data[j] = data[j].to(cuda)
                    # if args.use_cont_node_attr:
                    #     data[0] = norm_features(data[0])
                    optimizer.zero_grad()
                    output = model(data)
                    if len(output.shape) == 1:
                        output = output.unsqueeze(0)
                    loss = loss_fn(output, data[4])
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    time_iter = time.time() - start
                    train_loss += loss.item() * len(output)
                    n_samples += len(output)
                local_weights.append(copy.deepcopy(model.state_dict()))

                if args.train_verbose and (epoch % args.log_every == 0 or epoch == args.train_epochs - 1):
                    print('Train Epoch: %d\tLoss: %.4f (avg: %.4f) \tsec/iter: %.2f' % (
                        epoch + 1, loss.item(), train_loss / n_samples, time_iter / (batch_id + 1)))


        global_model.to(cpu)
        global_weights = copy.deepcopy(local_weights[0])
        for key in global_weights.keys():
            for i in range(1, len(local_weights)):
                global_weights[key] += local_weights[i][key]
            global_weights[key] = torch.div(global_weights[key], len(local_weights))

        # update global weights
        global_model.load_state_dict(global_weights)

        if e == args.train_epochs - 1:
            global_model.eval()
            start = time.time()
            for i in range(args.num_client):
                test_loss, correct, n_samples = 0, 0, 0
                for batch_id, data in enumerate(loaders['client_{}'.format(i)]['test']):
                    for j in range(len(data)):
                        data[j] = data[j].to(cuda)
                    # if args.use_org_node_attr:
                    #     data[0] = norm_features(data[0])
                    output = model(data)
                    if len(output.shape) == 1:
                        output = output.unsqueeze(0)
                    loss = loss_fn(output, data[4], reduction='sum')
                    test_loss += loss.item()
                    n_samples += len(output)
                    pred = predict_fn(output)

                    correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()

                eval_acc = 100. * correct / n_samples
                print('Client %d, Test set (epoch %d): Average loss: %.4f, Accuracy: %d/%d (%.2f%s) \tsec/iter: %.2f' % (
                    i, epoch + 1, test_loss / n_samples, correct, n_samples,
                    eval_acc, '%', (time.time() - start) / len(loaders['test'])))

        if args.save_clean_model:
            save_path = args.clean_model_save_path
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, '%s-%s-%s.t7' % (args.model, args.dataset, str(args.train_ratio)))

            torch.save({
                'model': global_model.state_dict(),
                'lr': args.lr,
                'batch_size': args.batch_size,
            }, save_path)
            print('Clean trained GNN saved at: ', os.path.abspath(save_path))

    return dr, model


if __name__ == '__main__':
    args = parse_args()
    run(args)

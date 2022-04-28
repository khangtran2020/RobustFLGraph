import sys, os
sys.path.append(os.path.abspath('../..'))
import os
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
sys.path.append(os.path.abspath('../..'))

import copy
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from utils.datareader import GraphData, DataReader
from torch.utils.data import DataLoader
from utils.batch import collate_batch
from utils.bkdcdd import select_cdd_graphs, select_cdd_nodes
from utils.mask import gen_mask, recover_mask
import main.benign as benign
import trojan.GTA as gta
from trojan.input import gen_input
from trojan.prop import train_model, evaluate
from config import parse_args
from model.gcn import GCN
from model.sage import GraphSAGE
from collections import OrderedDict


class GraphBackdoor:
    def __init__(self, args) -> None:
        self.args = args
        self.dr = DataReader(args)
        assert torch.cuda.is_available(), 'no GPU available'
        self.cpu = torch.device('cpu')
        self.cuda = torch.device('cuda')
        compromised_client = []
        for i in range(args.num_client):
            if self.dr.data['client_{}'.format(i)]['compromised'] == True:
                compromised_client.append(i)
        self.cp_client = compromised_client
        compromised_gid = []
        for i in self.cp_client:
            compromised_gid.extend(self.dr.data['client_{}'.format(i)]['train'])
        self.cp_gid_tr = compromised_gid
        compromised_gid = []
        for i in self.cp_client:
            compromised_gid.extend(self.dr.data['client_{}'.format(i)]['test'])
        self.cp_gid_te = compromised_gid

    def run(self):
        loaders = {}
        # pc_trim = 0.05
        threshold = args.rlr_thres
        for i in range(args.num_client):
            loaders['client_{}'.format(i)] = {}
            for split in ['train', 'test']:
                if split == 'train':
                    gids = self.dr.data['client_{}'.format(i)]['train']
                else:
                    gids = self.dr.data['client_{}'.format(i)]['test']
                gdata = GraphData(self.dr, gids)
                loader = DataLoader(gdata,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    collate_fn=collate_batch)
                # data in loaders['train/test'] is saved as returned format of collate_batch()
                loaders['client_{}'.format(i)][split] = loader
            print('train %d, test %d' % (
                len(loaders['client_{}'.format(i)]['train'].dataset),
                len(loaders['client_{}'.format(i)]['test'].dataset)))
        in_dim = loaders['client_{}'.format(0)]['train'].dataset.num_features
        out_dim = loaders['client_{}'.format(0)]['train'].dataset.num_classes
        models = {}
        if args.model == 'gcn':
            global_model = GCN(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
            for i in range(args.num_client):
                if i not in self.cp_client:
                    models['client_{}'.format(i)] = GCN(in_dim, out_dim, hidden_dim=args.hidden_dim,
                                                        dropout=args.dropout)
            cp_model = GCN(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
        elif args.model == 'sage':
            global_model = GraphSAGE(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
            for i in range(args.num_client):
                if i not in self.cp_client:
                    models['client_{}'.format(i)] = GraphSAGE(in_dim, out_dim, hidden_dim=args.hidden_dim,
                                                              dropout=args.dropout)
            cp_model = GraphSAGE(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
        else:
            raise NotImplementedError(args.model)

        nodenums = [adj.shape[0] for adj in self.dr.data['adj_list']]
        nodemax = max(nodenums)
        featdim = np.array(self.dr.data['features'][0]).shape[1]

        # init two generators for topo/feat
        toponet = gta.GraphTrojanNet(nodemax, self.args.gtn_layernum)
        featnet = gta.GraphTrojanNet(featdim, self.args.gtn_layernum)

        global_weights = global_model.state_dict()
        train_params = {}
        for i in range(args.num_client):
            if (i in self.cp_client):
                continue
            train_params['client_{}'.format(i)] = list(
                filter(lambda p: p.requires_grad, models['client_{}'.format(i)].parameters()))
        # print('N trainable parameters:', np.sum([p.numel() for p in train_params]))

        predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()
        loss_fn = F.cross_entropy
        # training
        glb_optimizer = optim.Adam(list(
            filter(lambda p: p.requires_grad, global_model.parameters())), lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.5, 0.999))
        global_model.to(self.cuda)
        global_model.train()

        for e in range(args.fl_epochs):
            print("FL update step {}".format(e))
            local_grad = []
            for i in range(args.num_client):
                if (i in self.cp_client):
                    continue
                model = models['client_{}'.format(i)]
                model.load_state_dict(global_weights)
                optimizer = optim.Adam(train_params['client_{}'.format(i)], lr=args.lr, weight_decay=args.weight_decay,
                                       betas=(0.5, 0.999))
                scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)
                model.to(self.cuda)
                for epoch in range(args.train_epochs):
                    model.train()
                    start = time.time()
                    train_loss, n_samples = 0, 0
                    for batch_id, data in enumerate(loaders['client_{}'.format(i)]['train']):
                        for j in range(len(data)):
                            data[j] = data[j].to(self.cuda)
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

                    if args.train_verbose and (epoch % args.log_every == 0 or epoch == args.train_epochs - 1):
                        print('Client %d, Train Epoch: %d\tLoss: %.4f (avg: %.4f) \tsec/iter: %.2f' % (i,
                                                                                                       epoch + 1,
                                                                                                       loss.item(),
                                                                                                       train_loss / n_samples,
                                                                                                       time_iter / (
                                                                                                               batch_id + 1)))
                model.to(self.cpu)
                local_weight = model.state_dict()
                local_grad.append(
                    OrderedDict({k: (global_weights[k] - local_weight[k]) for k in global_weights.keys()}))

            # backdoor
            cp_model.load_state_dict(global_weights)
            # bkd_gids_test, bkd_nids_test, bkd_nid_groups_test = self.bkd_cdd('test')
            cp_train_params = list(filter(lambda p: p.requires_grad, cp_model.parameters()))
            optimizer = optim.Adam(cp_train_params, lr=self.args.lr, weight_decay=self.args.weight_decay)

            for rs_step in range(self.args.resample_steps):  # for each step, choose different sample

                # randomly select new graph backdoor samples
                bkd_gids_train, bkd_nids_train, bkd_nid_groups_train = self.bkd_cdd('train')

                # positive/negtive sample set
                pset = bkd_gids_train
                nset = list(set(self.cp_gid_tr) - set(pset))

                if self.args.pn_rate != None:
                    if len(pset) > len(nset):
                        repeat = int(np.ceil(len(pset) / (len(nset) * self.args.pn_rate)))
                        nset = list(nset) * repeat
                    else:
                        repeat = int(np.ceil((len(nset) * self.args.pn_rate) / len(pset)))
                        pset = list(pset) * repeat

                # init train data
                # NOTE: for data that can only add perturbation on features, only init the topo value
                init_dr_train = self.init_trigger(
                    self.args, copy.deepcopy(self.dr), bkd_gids_train, bkd_nid_groups_train, 0.0, 0.0)
                bkd_dr_train = copy.deepcopy(init_dr_train)

                topomask_train, featmask_train = gen_mask(
                    init_dr_train, bkd_gids_train, bkd_nid_groups_train)
                Ainput_train, Xinput_train = gen_input(self.args, init_dr_train, bkd_gids_train)
                for bi_step in range(self.args.bilevel_steps):
                    print("Resampling step %d, bi-level optimization step %d" % (rs_step, bi_step))

                    toponet, featnet = gta.train_gtn(
                        self.args, cp_model, toponet, featnet,
                        pset, nset, topomask_train, featmask_train,
                        init_dr_train, bkd_dr_train, Ainput_train, Xinput_train)
                    # get new backdoor datareader for training based on well-trained generators
                    for gid in bkd_gids_train:
                        rst_bkdA = toponet(
                            Ainput_train[gid], topomask_train[gid], self.args.topo_thrd,
                            self.cpu, self.args.topo_activation, 'topo')
                        # rst_bkdA = recover_mask(nodenums[gid], topomask_train[gid], 'topo')
                        # bkd_dr_train.data['adj_list'][gid] = torch.add(rst_bkdA, init_dr_train.data['adj_list'][gid])
                        bkd_dr_train.data['adj_list'][gid] = torch.add(
                            rst_bkdA[:nodenums[gid], :nodenums[gid]].detach().cpu(),
                            init_dr_train.data['adj_list'][gid])

                        rst_bkdX = featnet(
                            Xinput_train[gid], featmask_train[gid], self.args.feat_thrd,
                            self.cpu, self.args.feat_activation, 'feat')
                        # rst_bkdX = recover_mask(nodenums[gid], featmask_train[gid], 'feat')
                        # bkd_dr_train.data['features'][gid] = torch.add(rst_bkdX, init_dr_train.data['features'][gid])
                        bkd_dr_train.data['features'][gid] = torch.add(
                            rst_bkdX[:nodenums[gid]].detach().cpu(), init_dr_train.data['features'][gid])

                    cp_model = train_model(self.args, bkd_dr_train, cp_model, list(set(pset)), list(set(nset)))
            cp_model.to(self.cpu)
            local_weight = cp_model.state_dict()
            local_grad.append(OrderedDict({k: (global_weights[k] - local_weight[k]) for k in global_weights.keys()}))
            # local_weights.append(copy.deepcopy(cp_model.state_dict()))
            """
                Update global model
            """
            global_grads = []
            mask_all = []
            for hlayer in local_grad[0].keys():
                parameters_h = [torch.flatten(t[hlayer]) for t in local_grad]
                parameters_h = torch.stack(parameters_h)
                parameters_h_sign = torch.sign(parameters_h)
                parameters_h_sum = sum(parameters_h_sign)
                mask = torch.where(parameters_h_sum >= threshold, torch.ones_like(parameters_h_sum),
                                   -torch.ones_like(parameters_h_sum))
                mask_all.append(mask)
                global_grads.append(torch.mul(torch.div(parameters_h_sum, 100), mask).view(
                    local_grad[0][hlayer].size()))

            global_model.to(self.cpu)
            for p, g in zip(global_model.parameters(), global_grads):
                p.grad = g
            glb_optimizer.step()

            # global_weights = copy.deepcopy(local_weights[0])
            # for key in global_weights.keys():
            #     for i in range(1, len(local_weights)):
            #         global_weights[key] += local_weights[i][key]
            #     global_weights[key] = torch.div(global_weights[key], len(local_weights))

            # update global weights
            global_model.load_state_dict(global_weights)

            if e == args.fl_epochs - 1:
                cp_model.load_state_dict(global_weights)
                bkd_gids_test, bkd_nids_test, bkd_nid_groups_test = self.bkd_cdd('test')
                init_dr_test = self.init_trigger(
                    self.args, copy.deepcopy(self.dr), bkd_gids_test, bkd_nid_groups_test, 0.0, 0.0)
                bkd_dr_test = copy.deepcopy(init_dr_test)

                topomask_test, featmask_test = gen_mask(
                    init_dr_test, bkd_gids_test, bkd_nid_groups_test)
                Ainput_test, Xinput_test = gen_input(self.args, init_dr_test, bkd_gids_test)
                for gid in bkd_gids_test:
                    rst_bkdA = toponet(
                        Ainput_test[gid], topomask_test[gid], self.args.topo_thrd,
                        self.cpu, self.args.topo_activation, 'topo')
                    # rst_bkdA = recover_mask(nodenums[gid], topomask_test[gid], 'topo')
                    # bkd_dr_test.data['adj_list'][gid] = torch.add(rst_bkdA,
                    #     torch.as_tensor(copy.deepcopy(init_dr_test.data['adj_list'][gid])))
                    bkd_dr_test.data['adj_list'][gid] = torch.add(
                        rst_bkdA[:nodenums[gid], :nodenums[gid]],
                        torch.as_tensor(copy.deepcopy(init_dr_test.data['adj_list'][gid])))

                    rst_bkdX = featnet(
                        Xinput_test[gid], featmask_test[gid], self.args.feat_thrd,
                        self.cpu, self.args.feat_activation, 'feat')
                    # rst_bkdX = recover_mask(nodenums[gid], featmask_test[gid], 'feat')
                    # bkd_dr_test.data['features'][gid] = torch.add(
                    #     rst_bkdX, torch.as_tensor(copy.deepcopy(init_dr_test.data['features'][gid])))
                    bkd_dr_test.data['features'][gid] = torch.add(
                        rst_bkdX[:nodenums[gid]], torch.as_tensor(copy.deepcopy(init_dr_test.data['features'][gid])))

                # graph originally in target label
                yt_gids = [gid for gid in bkd_gids_test
                           if self.dr.data['labels'][gid] == self.args.target_class]
                # graph originally notin target label
                yx_gids = list(set(bkd_gids_test) - set(yt_gids))
                clean_graphs_test = list(set(set(self.cp_gid_te)) - set(bkd_gids_test))

                # feed into GNN, test success rate
                bkd_acc = evaluate(self.args, bkd_dr_test, model, bkd_gids_test)
                flip_rate = evaluate(self.args, bkd_dr_test, cp_model, yx_gids)
                clean_acc = evaluate(self.args, bkd_dr_test, cp_model, clean_graphs_test)
                print("Backdoor Accuracy: {}, Clean Accuracy: {}, Flip Rate: {}".format(bkd_acc, flip_rate, clean_acc))
                if self.args.save_bkd_model:
                    save_path = self.args.bkd_model_save_path
                    os.makedirs(save_path, exist_ok=True)
                    save_path = os.path.join(save_path, '%s-%s-%f.t7' % (
                        self.args.model, self.args.dataset, self.args.train_ratio,
                        self.args.bkd_gratio_trainset, self.args.bkd_num_pergraph, self.args.bkd_size))

                    torch.save({'model': model.state_dict(),
                                'asr': bkd_acc,
                                'flip_rate': flip_rate,
                                'clean_acc': clean_acc,
                                }, save_path)
                    print("Trojaning model is saved at: ", save_path)

    def bkd_cdd(self, subset: str):
        # - subset: 'train', 'test'
        # find graphs to add trigger (not modify now)
        compromised_gid = []
        for i in self.cp_client:
            compromised_gid.extend(self.dr.data['client_{}'.format(i)][subset])
        bkd_gids = select_cdd_graphs(
            self.args, compromised_gid, self.dr.data['adj_list'], subset)
        # find trigger nodes per graph
        # same sequence with selected backdoored graphs
        bkd_nids, bkd_nid_groups = select_cdd_nodes(
            self.args, bkd_gids, self.dr.data['adj_list'])

        assert len(bkd_gids) == len(bkd_nids) == len(bkd_nid_groups)

        return bkd_gids, bkd_nids, bkd_nid_groups

    @staticmethod
    def init_trigger(args, dr: DataReader, bkd_gids: list, bkd_nid_groups: list, init_edge: float, init_feat: float):
        if init_feat == None:
            init_feat = - 1
            print('init feat == None, transferred into -1')

        # (in place) datareader trigger injection
        for i in tqdm(range(len(bkd_gids)), desc="initializing trigger..."):
            gid = bkd_gids[i]
            for group in bkd_nid_groups[i]:
                # change adj in-place
                src, dst = [], []
                for v1 in group:
                    for v2 in group:
                        if v1 != v2:
                            src.append(v1)
                            dst.append(v2)
                a = np.array(dr.data['adj_list'][gid])
                a[src, dst] = init_edge
                dr.data['adj_list'][gid] = a.tolist()

                # change features in-place
                featdim = len(dr.data['features'][0][0])
                a = np.array(dr.data['features'][gid])
                a[group] = np.ones((len(group), featdim)) * init_feat
                dr.data['features'][gid] = a.tolist()

            # change graph labels
            assert args.target_class is not None
            dr.data['labels'][gid] = args.target_class

        return dr


if __name__ == '__main__':
    args = parse_args()
    attack = GraphBackdoor(args)
    attack.run()


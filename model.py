from os.path import join

from networks import base_srl
from torch import nn
import torch
from torch.nn import functional as F


def get_model(args):
    if args.modality == 'fusion' or args.modality == 'test':
        rgb_model = base_srl.Ant_Model(args.num_class, args.verb_num_class, args.noun_num_class, 1024, 1024,
                                       args.embedding_dim, args.S_enc, args.S_ant, args.ant_dropout, args.enc_dropout,
                                       args.clc_dropout)
        #flow_model = base_srl.Ant_Model(args.num_class, args.verb_num_class, args.noun_num_class, 1024, 1024,
        #                                args.embedding_dim, args.S_enc, args.S_ant, args.ant_dropout, args.enc_dropout,
        #                                args.clc_dropout)
        obj_model = base_srl.Ant_Model(args.num_class, args.verb_num_class, args.noun_num_class, 352, 352,
                                       args.embedding_dim, args.S_enc, args.S_ant, args.ant_dropout, args.enc_dropout,
                                       args.clc_dropout)
        exp_name = args.exp_name.replace('_fusion', '_rgb')
        rgb_chk = torch.load(
            join(args.path_to_results, args.dataset, args.model_name, args.path_to_fusion_RGB, args.resume_timestamp,
                 exp_name + '_best.pth.tar'), weights_only=False)
        args.Fusion_weight.append(rgb_chk['train_perf'])

        #exp_name = args.exp_name.replace('_fusion', '_flow')
        #flow_chk = torch.load(
        #    join(args.path_to_results, args.dataset, args.model_name, args.path_to_fusion_FLOW, args.resume_timestamp,
        #         exp_name + '_best.pth.tar'), weights_only=False)
        #args.Fusion_weight.append(flow_chk['train_perf'])

        exp_name = args.exp_name.replace('_fusion', '_obj')
        obj_chk = torch.load(
            join(args.path_to_results, args.dataset, args.model_name, args.path_to_fusion_OBJ, args.resume_timestamp,
                 exp_name + '_best.pth.tar'), weights_only=False)
        args.Fusion_weight.append(obj_chk['train_perf'])

        rgb_model.load_state_dict(rgb_chk['state_dict'])
        #flow_model.load_state_dict(flow_chk['state_dict'])
        obj_model.load_state_dict(obj_chk['state_dict'])

        # model = [rgb_model, flow_model, obj_model]
        # model = FDHPFusion([rgb_model, flow_model], 2513, 0.5)
        model = FDHPFusion([rgb_model, obj_model], args.num_class, 0.5)
    elif args.modality != 'fusion':
        model = base_srl.Ant_Model(args.num_class, args.verb_num_class, args.noun_num_class, args.feat_in, args.hidden,
                                   args.embedding_dim, args.S_enc, args.S_ant, args.ant_dropout, args.enc_dropout,
                                   args.clc_dropout, 0.2, args.pre_train)

    return model

class FDHPFusion(nn.Module):
    def __init__(self, branches, hidden, dropout=0.8):
        """
            branches: list of pre-trained branches. Each branch should have the "return_context" property to True
            hidden: size of hidden vectors of the branches
            dropout: dropout probability
        """
        super(FDHPFusion, self).__init__()
        for model in branches:
            for param in model.parameters():
                param.requires_grad = False
        self.branches = nn.ModuleList(branches)
        # input size for the MATT network
        # given by 2 (hidden and cell state) * num_branches * hidden_size
        in_size = len(branches) * hidden
        # MATT network: an MLP with 3 layers
        self.MATT = nn.Sequential(nn.Linear(in_size, int(in_size / 4)),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(int(in_size / 4), int(in_size / 8)),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(int(in_size / 8), hidden))

    def forward(self, inputs):
        """inputs: tuple containing the inputs to the single branches"""
        preds_list = []
        contexts = []
        for i in range(len(inputs)):
            preds_t = self.branches[i](inputs[i])
            preds_list.append(preds_t)
            contexts.append(preds_t)

        context = torch.cat(contexts, 2)
        context = context.view(-1, context.shape[-1])
        matt_output = self.MATT(context)

        fused_preds = torch.zeros_like(preds_list[0])
        for i, preds in enumerate(preds_list):
            reshaped_matt_output = matt_output.view(preds.shape)
            fused_preds += preds + reshaped_matt_output

        return fused_preds

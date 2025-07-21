import torch
import torch.nn as nn
from torch.nn.init import normal, constant
from torch.nn import functional as F
from torch.nn import Parameter
from torch.nn import init
from torch import Tensor
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self-attention
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feedforward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class CustomEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers=2, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        return self.norm(output)

class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, query_pos=None):
        # Add query positional embedding
        tgt = tgt + query_pos if query_pos is not None else tgt
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # Cross-attention
        tgt2, _ = self.cross_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # Feedforward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class CustomDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers=2, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, query_pos=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, query_pos)
        return self.norm(output)

class FeaturePredictorTransformer(nn.Module):
    def __init__(self, feat_in, d_model=512, nhead=8, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.feat_in = feat_in
        self.d_model = d_model

        # Project input features to d_model
        self.input_proj = nn.Linear(feat_in, d_model)
        nn.init.kaiming_normal_(self.input_proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.input_proj.bias, 0)

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, 100, d_model))  # Max sequence length 100
        self.query_pos = nn.Parameter(torch.randn(1, 1, d_model))  # Query for next frame

        # Encoder
        self.encoder = CustomEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        # Decoder
        self.decoder = CustomDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, feat_in)
        nn.init.kaiming_normal_(self.output_proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, inputs):
        # inputs: [B, L, S]
        B, L, S = inputs.shape

        # Project inputs
        src = self.input_proj(inputs)  # [B, L, d_model]

        # Add positional embeddings
        pos_embed = self.pos_embed[:, :L, :].expand(B, -1, -1)  # [B, L, d_model]
        src = src + pos_embed

        # Encode
        memory = self.encoder(src)  # [B, L, d_model]

        # Decoder query
        tgt = torch.zeros(B, 1, self.d_model, device=inputs.device)  # [B, 1, d_model]
        query_pos = self.query_pos.expand(B, -1, -1)  # [B, 1, d_model]

        # Decode
        output = self.decoder(tgt, memory, query_pos)  # [B, 1, d_model]

        # Project to output
        output = self.output_proj(output.squeeze(1))  # [B, feat_in]

        return output

class EncGRU(nn.Module):
    def __init__(self, feat_in, feat_out, num_layers=1, dropout=0):
        super(EncGRU, self).__init__()
        self.gru = nn.GRU(feat_in, feat_out, num_layers=num_layers, dropout=dropout)

    def forward(self, seq):
        last_hid = None
        hid = []
        for i in range(seq.shape[0]):
            el = seq[i, ...].unsqueeze(0)
            if last_hid is not None:
                _, last_hid = self.gru(el, last_hid)
            else:
                _, last_hid = self.gru(el)
            hid.append(last_hid)
        return torch.stack(hid, 0)

class Ant_Model(nn.Module):
    def __init__(self, num_class, verb_num_class, noun_num_class, feat_in, hidden, embedding_dim, S_enc, S_ant, ant_dropout_ratio=0.8, enc_dropout_ratio=0.8, clc_dropout_ratio=0.8, similarity_threshold = 0, pre_train=False, depth=1):
        super(Ant_Model, self).__init__()

        ##  parameters
        self.input_size = feat_in
        self.hidden_size = hidden
        self.embedding_dim = embedding_dim
        self.S_enc = S_enc
        self.S_ant = S_ant
        self.enc_gru = EncGRU(self.input_size, self.hidden_size, num_layers=depth)
        self.ant_dropout = nn.Dropout(ant_dropout_ratio)
        self.enc_dropout = nn.Dropout(enc_dropout_ratio)
        self.clc_dropout = nn.Dropout(clc_dropout_ratio)
        self.fus_dropout = nn.Dropout(clc_dropout_ratio)
        d_model = 256 if feat_in == 1024 else 128
        self.Ant_Model_Transformer = FeaturePredictorTransformer(feat_in, d_model=d_model, nhead=8)

        ## ant gru
        self.input_size_a = self.hidden_size + self.input_size
        self.hidden_size_a = self.hidden_size
        self.ant_gru = nn.GRU(self.input_size_a, self.hidden_size_a, num_layers=1)
        self.similarity_threshold = similarity_threshold
        self.pre_train = pre_train
        if self.pre_train is False:
            self.classifier_hidden_size = self.hidden_size * 2
            for param in self.Ant_Model_Transformer.parameters():
                param.requires_grad = False
        else:
            self.classifier_hidden_size = self.hidden_size
        ## prediction

        self.classifier = nn.Linear(self.classifier_hidden_size, num_class)
        self.classifier_FGM = nn.Linear(self.hidden_size, num_class)
        ## init parameters
        for l in self.children():
           if isinstance(l, nn.Linear):
                nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(l.bias, 0)
    def forward(self, inputs):
        if self.pre_train:
            return self.forward_pre_model(inputs)
        else:
            return self.forward_model(inputs)
    def forward_model(self, inputs):

        inputs = inputs.permute(1, 0, 2)  # batch * length * dim --> length * batch * dim
        temp_inputs = self.ant_dropout(F.relu(inputs))
        features = self.enc_gru(temp_inputs)
        features = features.squeeze(1).contiguous()  # length * batch * dim

        # accumulate the predictions in a list
        feature_predictions = []
        predictions = []

        # for each time-step
        for t in range(features.shape[0]):

            ## init input
            i_t = torch.cat((inputs[t, ...], features[t, ...]), dim=1)
            i_t = self.enc_dropout(F.relu(i_t)).unsqueeze(0)
            h_t = features[t, ...].unsqueeze(0)


            ant_length = inputs.shape[0] - t
            for index in range(ant_length):
                ## ant gru
                # update gate
                _, h_t = self.ant_gru(i_t, h_t)

                ## reattend
                query_feature = h_t.permute(1, 0, 2)
                reattend_feature = self.Ant_Model_Transformer(query_feature).squeeze(1)
                # update gate
                query_feature = query_feature.squeeze(1)
                temp_h_t_re = reattend_feature.squeeze(0)
                ## next inputs

                i_t = torch.cat((temp_h_t_re, features[t, ...]), dim=1)
                i_t = self.enc_dropout(F.relu(i_t)).unsqueeze(0)
                ## clc feature
                clc_feature = torch.cat((query_feature, temp_h_t_re), dim=1)
                # accumulate
            feature_predictions.append(temp_h_t_re)

            # accumulate
            predictions.append(clc_feature)

        feature_predictions = torch.stack(feature_predictions, 1)
        x = torch.stack(predictions, 1)
        temp_x = x.view(-1, x.size(2))
        temp_FGM = feature_predictions.view(-1, feature_predictions.size(2))

        ## classifier
        temp_x = self.clc_dropout(F.relu(temp_x))
        temp_FGM = self.clc_dropout(F.relu(temp_FGM))
        y = self.classifier(temp_x).view(x.size(0), x.size(1), -1)
        fgm = self.classifier_FGM(temp_FGM).view(feature_predictions.size(0), feature_predictions.size(1), -1)
        return y, fgm
    def forward_pre_model(self, inputs):
        B, L, S = inputs.shape

        # Step 1: Insert virtual frames
        inputs, insert_positions = self.insert_virtual_frames(inputs)  # [B, L_new, S], List[List[int]]

        # Permute for GRU
        inputs = inputs.permute(1, 0, 2)  # [L_new, B, S]

        # Apply dropout and GRU
        temp_inputs = self.ant_dropout(F.relu(inputs))
        features = self.enc_gru(temp_inputs)  # [L_new, B, hidden_size]
        features = features.squeeze(1).contiguous()  # [L_new, B, hidden_size]

        # Remove virtual frames
        features = self.remove_virtual_frames(features, insert_positions, L)  # [L, B, hidden_size]
        inp = features[-1:].permute(1, 0, 2)
        predictions = self.Ant_Model_Transformer(inp)
        predictions = self.classifier(predictions)
        return predictions, None

    def compute_similarity(self, inputs):
        # inputs: [B, L, S]
        B, L, S = inputs.shape
        # Normalize features for cosine similarity
        norm_inputs = F.normalize(inputs, p=2, dim=-1)  # [B, L, S]
        # Compute cosine similarity between adjacent frames
        sim = torch.bmm(norm_inputs[:, :-1, :], norm_inputs[:, 1:, :].transpose(1, 2)).diagonal(0, 1, 2)  # [B, L-1]
        return sim

    def generate_virtual_frame(self, frame1, frame2):
        # Linear interpolation: frame1, frame2: [B, S]
        return (frame1 + frame2) / 2.0

    def insert_virtual_frames(self, inputs):
        # inputs: [B, L, S]
        B, L, S = inputs.shape
        sim = self.compute_similarity(inputs)  # [B, L-1]
        insert_mask = sim < self.similarity_threshold  # [B, L-1]
        new_frames = []
        insert_positions = []  # List of lists: positions per batch

        for b in range(B):
            batch_frames = [inputs[b, 0, :]]  # Start with first frame
            batch_positions = []
            for t in range(L - 1):
                batch_frames.append(inputs[b, t + 1, :])
                if insert_mask[b, t]:
                    # Insert one virtual frame
                    virtual_frame = self.generate_virtual_frame(inputs[b, t, :], inputs[b, t + 1, :])
                    batch_frames.insert(-1, virtual_frame)
                    batch_positions.append(t + 1 + len(batch_positions))
            new_frames.append(torch.stack(batch_frames))  # [L_new_b, S]
            insert_positions.append(batch_positions)

        # Find max new length
        max_L_new = max(len(frames) for frames in new_frames)
        # Pad to uniform length
        padded_frames = torch.zeros(B, max_L_new, S, device=inputs.device)
        for b in range(B):
            padded_frames[b, :len(new_frames[b]), :] = new_frames[b]

        return padded_frames, insert_positions  # [B, L_new, S], List[List[int]]

    def remove_virtual_frames(self, features, insert_positions, original_L):
        # features: [L_new, B, hidden_size]
        # insert_positions: List[List[int]], positions in [0, L_new-1]
        # original_L: Original sequence length
        B = features.shape[1]
        L_new = features.shape[0]
        new_features = []

        for b in range(B):
            # Indices to keep (exclude insert_positions)
            keep_indices = [i for i in range(L_new) if i not in insert_positions[b]]
            # Ensure we keep exactly original_L frames
            if len(keep_indices) != original_L:
                keep_indices = keep_indices[:original_L]  # Truncate if needed
            batch_features = features[keep_indices, b, :]  # [original_L, hidden_size]
            new_features.append(batch_features)

        return torch.stack(new_features).permute(1, 0, 2)  # [original_L, B, hidden_size]


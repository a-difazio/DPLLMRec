import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                         args.num_heads,
                                                         args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):  # TODO: fp64 and int64 as default in python, trim?
        # item embeddings
        # seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5

        # positional embedding
        # poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        batch_size, seq_len = log_seqs.shape
        poss = torch.arange(1, seq_len + 1, device=self.dev).unsqueeze(0).repeat(batch_size, 1)
        poss *= (log_seqs != 0)

        # combine
        #seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        # attention mask
        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        # for each block
        for i in range(len(self.attention_layers)):
            # (batch_size, maxlen, hidden_units) -> (maxlen, batch_size, hidden_units)
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                # pre-normalization
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x,
                                                attn_mask=attention_mask)
                seqs = seqs + mha_outputs  # add residuals
                # (maxlen, batch_size, hidden_units) -> (batch_size, maxlen, hidden_units)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))  # residual + FFN
            else:
                # post-normalization (standard transformer)
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
                                                attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs) # add residual + norm
                # (maxlen, batch_size, hidden_units) -> (batch_size, maxlen, hidden_units)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))  # residual + FFN + norm

        # (batch_size, maxlen, hidden_units) -> (batch_size, maxlen, hidden_units)
        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats


    def forward(self, user_ids, log_seqs):
        # se volessi aggiungere l'embedding utente
        # user_emb = self.user_emb(user_ids)  # (B, C)
        # user_emb = user_emb.unsqueeze(1)    # (B, 1, C)
        # log_feats = log_feats + user_emb    # broadcasting su T

        log_feats = self.log2feats(log_seqs)
        logits = log_feats @ self.item_emb.weight.T

        return logits

    def predict(self, user_ids, log_seqs):

        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        logits = final_feat @ self.item_emb.weight.T

        return logits

"""
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        # for training, sequence, positive items, negative items
        # (batch_size, maxlen, hidden_units), contextual representation (features)
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        # (batch_size, maxlen), items after log_seqs -> (batch_size, maxlen, hidden_units)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        # (batch_size, maxlen), negative sampling -> (batch_size, maxlen, hidden_units)
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        # (batch_size, maxlen), item relevance
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        # (batch_size, maxlen), item relevance
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits  # pos_pred, neg_pred
"""

"""
    def predict(self, user_ids, log_seqs, item_indices):
        # for inference, sequence, sequence of candidates items ID
        # (batch_size, maxlen, hidden_units), contextual representation (features)
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        # (batch_size, hidden_units), extract feature vector for the last item in the sequence, for each batch
        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste
        # alternatives:
        # - pooling/aggregation: mean, sum o max-pooling, of all log_feats for the sequence
        # - output attention

        # (batch_size, num_candidates) -> (batch_size, num_candidates, hidden_units), items for witch compute the scores
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)

        # for each batch (num_candidates, hidden_units) * (hidden_units, 1) -> (batch_size, num_candidates)
        # each logits[b, i] is the relevance score for user b, item i
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)
        
"""

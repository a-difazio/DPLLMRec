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

    @torch.no_grad()
    def generate(self, user, seq, config, options):

        context_len = options.context_len if options.context_len else min(config.maxlen, len(seq))
        temperature = options.temperature
        penalty = options.penalty
        no_repeat = options.no_repeat
        include_context = options.include_context
        top_k = options.top_k
        top_p = options.top_p

        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if top_k is not None and top_p is not None:
            raise ValueError("Use either top_k or top_p, not both.")

        # Prendo i primi context len items
        # prompt = seq[:context_len]
        # proviamo con approccio classico in cui prendo gli ultimi context len item
        prompt = seq[-context_len:]
        # Lunghezza della generazione pari al massimo (se specificato) o alla lunghezza originale della sequenza
        prompt_len = len(seq)
        gen_len = prompt_len
        generated_sequence = []

        # Counts delle generazioni per ogni item
        counts = torch.zeros(self.item_num + 1, dtype=torch.long, device=self.dev)

        for step in range(gen_len):
            # Padding
            # Inizializzo un tensore di maxlen per il padding
            prompt_tensor = torch.zeros(config.maxlen, dtype=torch.long, device=self.dev)
            # "taglio" il prompt a maxlen (prendendo la coda della sequenza)
            prompt_cut = prompt[-config.maxlen:]
            # Adesso riempio il tensore con la sequenza così preparata
            prompt_tensor[-len(prompt_cut):] = torch.tensor(prompt_cut, dtype=torch.long, device=self.dev)
            # questo non ho capito a cosa mi serve, ora controllo
            prompt_tensor = prompt_tensor.unsqueeze(0)

            # colcolo i logits
            logits = self.predict(user, prompt_tensor)
            # faccio masking sull'inidce 0 che è il padding
            logits[:, 0] = float('-inf')
            logits = logits / temperature

            # Penalità anti-repetition soft
            # applico una penalità sui logits, proporzionale a quante volte è già stato generato un item
            if not no_repeat and penalty > 0:
                # logits -= penalty * counts.float()
                repeated = counts > 0
                positive = logits > 0

                logits[repeated & positive] /= penalty
                logits[repeated & ~positive] *= penalty


            # Anti-repetition
            if no_repeat and penalty == 0:
                logits[:, generated_sequence] = float('-inf')

            #  Top-k sampling
            if top_k is not None and top_p is None:
                k = min(int(top_k), logits.size(-1))
                if k > 0:
                    topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
                    filtered_logits = torch.full_like(logits, float('-inf'))
                    filtered_logits.scatter_(dim=-1, index=topk_idx, src=topk_vals)
                    logits = filtered_logits

            # Top-p Sampling
            if top_p is not None and top_k is None:
                sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                remove = cumulative_probs > top_p
                remove[:, 0] = False

                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

                filtered_logits = torch.full_like(logits, float('-inf'))
                filtered_logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
                logits = filtered_logits

            # Guardrail: se tutti i logits sono invalidi, fallback uniforme sugli item validi (escluso padding).
            if not torch.isfinite(logits).any(dim=-1).all():
                logits = torch.zeros_like(logits)
                logits[:, 0] = float('-inf')

            # calcolo la probabilità di ogni item
            probs = torch.softmax(logits, dim=-1)

            # Guardrail numerico per evitare NaN/degenerate distribution.
            invalid_probs = (~torch.isfinite(probs)).any(dim=-1) | (probs.sum(dim=-1) <= 0)
            if invalid_probs.any():
                probs = torch.zeros_like(probs)
                probs[:, 1:] = 1.0
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # Sampling del next item
            next_item = torch.multinomial(probs, 1).item()

            # Appendo il next item alla sequenza degli item generati
            generated_sequence.append(next_item)
            # E al prompt
            prompt.append(next_item)

            # Aggiorno il counter
            counts[next_item] += 1
            assert counts[0] == 0, "Padding generated as next item!"

        return prompt if include_context else generated_sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import numpy as np




class MultiHeadedAttention_original(nn.Module):

    def __init__(self, head_count, model_dim, d_k, d_v, dropout=0.1,
                 max_relative_positions=0, use_neg_dist=True, coverage=False):
        super(MultiHeadedAttention_original, self).__init__()

        self.head_count = head_count
        self.model_dim = model_dim
        self.d_k = d_k
        self.d_v = d_v



        self.key = nn.Linear(model_dim, head_count * self.d_k)
        self.query = nn.Linear(model_dim, head_count * self.d_k)
        self.value = nn.Linear(model_dim, head_count * self.d_v)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.head_count * d_v, model_dim)
        self._coverage = coverage

        self.max_relative_positions = max_relative_positions
        self.use_neg_dist = use_neg_dist

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1 \
                if self.use_neg_dist else max_relative_positions + 1


            self.relative_positions_embeddings_k = nn.Embedding(
                vocab_size, self.d_k)
            self.relative_positions_embeddings_v = nn.Embedding(
                vocab_size, self.d_v)

    def forward(self, key, value, query, mask=None, layer_cache=None,
                attn_type=None, step=None, coverage=None):

        batch_size = key.size(0)
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)
        use_gpu = key.is_cuda

        def shape(x, dim):
            """  projection """
            return x.view(batch_size, -1, head_count, dim).transpose(1, 2)

        def unshape(x, dim):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                # 1) Project key, value, and query.
                key = shape(self.key(key), self.d_k)
                value = shape(self.value(value), self.d_v)
                query = shape(self.query(query), self.d_k)

                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"], key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"], value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value

            elif attn_type == "context":
                query = shape(self.query(query), self.d_k)
                if layer_cache["memory_keys"] is None:
                    key = shape(self.key(key), self.d_k)
                    value = shape(self.value(value), self.d_v)
                else:
                    key, value = layer_cache["memory_keys"], \
                                 layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = shape(self.key(key), self.d_k)
            value = shape(self.value(value), self.d_v)
            query = shape(self.query(query), self.d_k)

        if self.max_relative_positions > 0 and attn_type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions, self.use_neg_dist,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings_k(
                relative_positions_matrix.to(key.device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings_v(
                relative_positions_matrix.to(key.device))

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(self.d_k)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        if self.max_relative_positions > 0 and attn_type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        # ----------------------------
        # We adopt coverage attn described in Paulus et al., 2018
        # REF: https://arxiv.org/abs/1705.04304
        exp_score = None
        if self._coverage and attn_type == 'context':
            # batch x num_heads x query_len x 1
            maxes = torch.max(scores, 3, keepdim=True)[0]
            # batch x num_heads x query_len x key_len
            exp_score = torch.exp(scores - maxes)

            if step is not None:  # indicates inference mode (one-step at a time)
                if coverage is None:
                    # t = 1 in Eq(3) from Paulus et al., 2018
                    unnormalized_score = exp_score
                else:
                    # t = otherwise in Eq(3) from Paulus et al., 2018
                    assert coverage.dim() == 4  # B x num_heads x 1 x key_len
                    unnormalized_score = exp_score.div(coverage + 1e-20)
            else:
                multiplier = torch.tril(torch.ones(query_len - 1, query_len - 1))
                # batch x num_heads x query_len-1 x query_len-1
                multiplier = multiplier.unsqueeze(0).unsqueeze(0). \
                    expand(batch_size, head_count, *multiplier.size())
                multiplier = multiplier.cuda() if scores.is_cuda else multiplier

                # B x num_heads x query_len-1 x key_len
                penalty = torch.matmul(multiplier, exp_score[:, :, :-1, :])
                # B x num_heads x key_len
                no_penalty = torch.ones_like(penalty[:, :, -1, :])
                # B x num_heads x query_len x key_len
                penalty = torch.cat([no_penalty.unsqueeze(2), penalty], dim=2)
                assert exp_score.size() == penalty.size()
                unnormalized_score = exp_score.div(penalty + 1e-20)

            # Eq.(4) from Paulus et al., 2018
            attn = unnormalized_score.div(unnormalized_score.sum(3, keepdim=True))

        # Softmax to normalize attention weights
        else:
            # 3) Apply attention dropout and compute context vectors.
            attn = self.softmax(scores).to(query.dtype)

        # ----------------------------

        # 3) Apply attention dropout and compute context vectors.
        # attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)

        if self.max_relative_positions > 0 and attn_type == "self":
            context = unshape(context_original
                              + relative_matmul(drop_attn,
                                                relations_values,
                                                False),
                              self.d_v)
        else:
            context = unshape(context_original, self.d_v)

        final_output = self.output(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # a list of size num_heads containing tensors
        # of shape `batch x query_len x key_len`
        attn_per_head = [attn.squeeze(1)
                         for attn in attn.chunk(head_count, dim=1)]

        covrage_vector = None
        if (self._coverage and attn_type == 'context') and step is not None:
            covrage_vector = exp_score  # B x num_heads x 1 x key_len

        return final_output, attn_per_head, covrage_vector

    def update_dropout(self, dropout):
        self.dropout.p = dropout




def attention(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0                             
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def clones(module, N):  
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        x = x.float()
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class LocalRNN(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_type, ksize):
        super(LocalRNN, self).__init__()
        
        """
        LocalRNN structure
        """
        self.ksize = ksize
        self.input_dim = input_dim
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(output_dim, output_dim, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_dim, output_dim, batch_first=True)
        elif rnn_type == 'Linear':
            self.rnn = nn.Linear(output_dim, output_dim)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(output_dim, output_dim, batch_first=True)
        elif rnn_type == 'atten':
            self.rnn = MultiHeadedAttention_original(4, output_dim,64,64,dropout= 0.2)
        else:
            raise NotImplementedError



    def forward(self, x):
        idx = [i for j in range(self.ksize - 1, 10000, 1) for i in range(j - (self.ksize - 1), j + 1, 1)]
        self.select_index = torch.LongTensor(idx).to(x.device)
        self.zeros = torch.zeros((self.ksize - 1, self.input_dim)).to(x.device)

        x = self.get_K(x)  # b x seq_len x ksize x d_model
        batch, l, ksize, d_model = x.shape
        if self.rnn_type == 'Linear': # 全连接层
            h = self.rnn(x.view(-1, self.ksize, d_model))[:, -1, :]
        elif self.rnn_type == 'atten':
            h = self.rnn(x.view(-1, self.ksize, d_model),x.view(-1, self.ksize, d_model),x.view(-1, self.ksize, d_model))[0][:, -1, :]
        else:
            h = self.rnn(x.view(-1, self.ksize, d_model))[0][:, -1, :]
        return h.view(batch, l, d_model)

    def get_K(self, x):
        self.zeros = torch.zeros((self.ksize - 1, self.input_dim)).to(x.device)

        batch_size, l, d_model = x.shape
        zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1)

        x = torch.cat((zeros, x), dim=1)
        key = torch.index_select(x, 1, self.select_index[:self.ksize * l]).to(x.device)

        key = key.reshape(batch_size, l, self.ksize, -1)
        return key


class SublayerConnection(nn.Module): 
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class LocalRNNLayer(nn.Module): 
    "Encoder is made up of attconv and feed forward (defined below)"

    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout):
        super(LocalRNNLayer, self).__init__()
        self.local_rnn = LocalRNN(input_dim, output_dim, rnn_type, ksize)
        self.connection = SublayerConnection(output_dim, dropout)

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.connection(x, self.local_rnn)
        return x


class LocalRNNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout, N=3):
        super(LocalRNNBlock, self).__init__()
        self.layers = clones(
            LocalRNNLayer(input_dim, output_dim, rnn_type, ksize, dropout), N)

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x

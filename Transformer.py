import torch
import torch.nn as nn

# this is deprecated, what's the best way to create a variable without gradients?
from torch.autograd import Variable
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_trg_mask(trg_len, device):

    lower_triangular = torch.tril(torch.ones(trg_len, trg_len))
    trg_mask = lower_triangular.unsqueeze(0).unsqueeze(0)

    return trg_mask.to(device)


def create_src_mask(src, src_pad_idx, device):

    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

    return src_mask.to(device)


# naive non-multi-head attention
class Attention(nn.Module):
    def __init__(self, embedding_dim):

        super(Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)

    # Q, K, V all have shape (batch_size, seq_len, emb_dim)
    def forward(self, query, key, value):

        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)
        attn_weights = torch.bmm(query, key.transpose(1, 2))
        attn_weight_softmax = torch.softmax(attn_weights / (self.embedding_dim ** 0.5), dim=2)

        out = torch.bmm(attn_weight_softmax, value)

        return out

# Two ways to implement attention:
# TODO: add an assert statement that says embedding dim must be divisible by number of heads

class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, heads):

        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.head_dim = embedding_dim // heads
        self.W_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.out_fc = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, query, key, value, mask=None):

        # Q, K, V all have shape (batch_size, {Q, K, V}_len, embedding_dim)
        batch_size = query.shape[0]

        query_len = query.shape[1]
        key_len = key.shape[1]
        value_len = value.shape[1]

        # each below have shape (batch_size, seq_len, heads, head_dim)
        query = torch.reshape(query, (batch_size, query_len, self.heads, self.head_dim))
        key = torch.reshape(key, (batch_size, key_len, self.heads, self.head_dim))
        value = torch.reshape(value, (batch_size, value_len, self.heads, self.head_dim))

        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        # shape: (batch_size, heads, query_len, key_len)
        # for each element in the batch, have a weight matrix per head
        attn_weights = torch.einsum('bqhd, bkhd -> bhqk', [query, key])

        #print(mask)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -float('inf'))

        attn_weights_softmax = torch.softmax(attn_weights / self.embedding_dim ** 0.5, dim = 3)

        # out = torch.einsum('bhqv, bvhd -> bhqd', [attn_weights_softmax, value])
        out = torch.einsum('bhqv, bvhd -> bqhd', [attn_weights_softmax, value])
        out = torch.reshape(out, (out.shape[0], out.shape[1], out.shape[2]*out.shape[3]))
        # out = torch.reshape(out, (out.shape[0], out.shape[2], out.shape[1]*out.shape[3]))
        out = self.out_fc(out)

        return out

class PositionalEncoder(nn.Module):

    def __init__(self, embedding_dimension, max_len=5000):
        super(PositionalEncoder, self).__init__()
        self.seq_len = max_len
        self.embedding_dimension = embedding_dimension

        positional_encoding = torch.zeros(self.seq_len, self.embedding_dimension)

        base_position_encoder = self.generate_outer_product(self.seq_len, self.embedding_dimension)
        positional_encoding[:, 0::2] = torch.sin(base_position_encoder)
        positional_encoding[:, 1::2] = torch.cos(base_position_encoder)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)

    @staticmethod
    def generate_outer_product(L, d_emb):

        # vertical
        sequence = torch.arange(0, L)

        # horizontal
        k = torch.arange(0, d_emb // 2).float()
        omega = torch.exp(-(2*k/d_emb)*math.log(10000.))

        return torch.einsum('i,j -> ij', sequence, omega).float()

    def forward(self, x):
        #print("positional encoder")
        #self.pe[:, :x.shape[1], :]
        #print(Variable(self.pe[:, :x.shape[1], :], requires_grad=False).shape)

        return x + self.positional_encoding[:, :x.shape[1], :]


class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, heads, multiplier, dropout):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embedding_dim, heads)
        self.layernorm_1 = nn.LayerNorm(embedding_dim)
        self.layernorm_2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*multiplier),
            nn.ReLU(),
            nn.Linear(embedding_dim*multiplier, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):

        attention_forward = self.attention(query, key, value, mask)

        # skip connection
        sub_block_1 = self.dropout(self.layernorm_1(query + attention_forward))

        # skip connection
        sub_block_2 = self.dropout(self.layernorm_2(sub_block_1 + self.feed_forward(sub_block_1)))

        return sub_block_2


class Encoder(nn.Module):

    def __init__(
            self,
            src_vocab_size,
            embedding_dim,
            max_len,
            num_layers,
            heads,
            multiplier,
            dropout
    ):

        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoder(embedding_dim, max_len)

        self.transformer_layers = nn.ModuleList(

            [
                TransformerBlock(
                    embedding_dim,
                    heads,
                    multiplier,
                    dropout
                )

                for _ in range(num_layers)

            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        out = self.embedding(x)
        out = self.positional_encoding(out)
        out = self.dropout(out)

        for layer in self.transformer_layers:

            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):

    def __init__(self,
                 embedding_dim,
                 heads,
                 multiplier,
                 dropout
    ):

        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embedding_dim, heads)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.transformer_block = TransformerBlock(embedding_dim, heads, multiplier, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key, value, src_mask=None, trg_mask=None):

        query = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.layer_norm(query + x))
        #out = self.transformer_block(value, key, query, src_mask)
        out = self.transformer_block(query, key, value, src_mask)

        return out


# Knock these out tomorrow evening, once and for all
class Decoder(nn.Module):

    def __init__(self,
                 trg_vocab_size,
                 embedding_dim,
                 max_len,
                 num_layers,
                 heads,
                 multiplier,
                 dropout
    ):

        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(trg_vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoder(embedding_dim, max_len)

        self.decoder_layers = nn.ModuleList(

            [
                DecoderBlock(
                    embedding_dim,
                    heads,
                    multiplier,
                    dropout
                )

                for _ in range(num_layers)

            ]

        )

        self.linear = nn.Linear(embedding_dim, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key, value, src_mask=None, trg_mask=None):

        # implement
        out = self.embedding(x)
        out = self.positional_encoding(out)
        out = self.dropout(out)

        for layer in self.decoder_layers:

            out = layer(out, key, value, src_mask, trg_mask)

        out = self.linear(out)


        return out


class Transformer(nn.Module):

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embedding_dim,
                 max_len,
                 heads,
                 num_layers,
                 multiplier,
                 dropout,
                 device
                 ):

        super(Transformer, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx


        self.encoder = Encoder(
            src_vocab_size,
            embedding_dim,
            max_len,
            num_layers,
            heads,
            multiplier,
            dropout
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embedding_dim,
            max_len,
            num_layers,
            heads,
            multiplier,
            dropout
        )
        
        self.device = device

    def create_src_mask(self, src, src_pad_idx):

        src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask.to(self.device)

    def create_trg_mask(self, trg_len):

        lower_triangular = torch.tril(torch.ones(trg_len, trg_len))
        trg_mask = lower_triangular.unsqueeze(0).unsqueeze(0)

        return trg_mask.to(self.device)


    def forward(self,
                src_batch,
                trg_batch,
                ):

        _, trg_seq_len = trg_batch.shape
        src_mask = self.create_src_mask(src_batch, self.src_pad_idx)
        trg_mask = self.create_trg_mask(trg_seq_len)

        encoded_src_batch = self.encoder(src_batch, src_mask)

        decoded_logits = self.decoder(trg_batch, encoded_src_batch, encoded_src_batch, src_mask, trg_mask)

        return decoded_logits


if __name__ == '__main__':

    transformer = Transformer(
        src_vocab_size=100,
        trg_vocab_size=110,
        src_pad_idx=0,
        trg_pad_idx=0,
        embedding_dim=256,
        max_len=50,
        heads=8,
        num_layers=6,
        multiplier=4,
        dropout=0.1,
        device=device

    ).to(device)

    src = torch.tensor([[1, 2, 3, 4, 3, 0],[5, 6, 8, 2, 5, 0]]).to(device)
    #trg = torch.tensor([[1, 7, 8, 0, 0, 0, 5], [1, 8, 7, 3, 2, 8, 5]])
    trg = torch.tensor([[1], [1]]).to(device)

    embedded_trg = torch.randn(2, 8, 128)

    print(src)
    print(trg)

    print(transformer(src, trg[:, :]).shape)

    """

    encoder = Encoder(100, 128, 40, 6, 8, 6, 0.2)
    decoder_block = DecoderBlock(128, 1, 4, 0.2)
    decoder = Decoder(110, 128, 40, 8, 4, 2, 0.2)


    Y = encoder(src)
    print(Y.shape)

    #out = decoder_block(embedded_trg, Y, Y)
    #print(out.shape)

    out = decoder(trg, Y, Y)
    print(encoder)
    print(out.shape)
    
    """

    """
    X = torch.randn(16, 10, 256)
    attention = MultiHeadAttention(256, 8)
    out = attention(X, X, X)
    print(out.shape)

    transformer_block = TransformerBlock(256, 8, 4, 0.3)

    transformer = transformer_block(X, X, X)
    print(transformer.shape)
    """


    """
    X = torch.zeros(1, 5, 10)
    print(X)
    position = PositionalEncoder(10)
    #print(position.pe)
    print(position(X))
    """







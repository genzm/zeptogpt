import torch
import torch.nn as nn
import torch.nn.functional as F


########## Attention and Transformer implementation ##########
def QKVattention(Query, Key, Value, attention_QKV_dim):
    attention_score = torch.matmul(Query, Key.transpose(-2, -1))/attention_QKV_dim
    attention_weight = F.softmax(attention_score, dim=-1)
    attention = torch.matmul(attention_weight, Value)
    return attention

class multihead_attention(nn.Module):
    def __init__(self, Q_tensor_dim, K_tensor_dim, V_tensor_dim, attention_QKV_dim) -> None:
        super().__init__()
        self.Q_tensor_dim = Q_tensor_dim
        self.K_tensor_dim = K_tensor_dim
        self.V_tensor_dim = V_tensor_dim
        self.attention_QKV_dim = attention_QKV_dim
        self.linear_Q = nn.Linear(Q_tensor_dim, attention_QKV_dim)
        self.linear_K = nn.Linear(K_tensor_dim, attention_QKV_dim)
        self.linear_V = nn.Linear(V_tensor_dim, attention_QKV_dim)

    def forward(self, Q_tensor, K_tensor, V_tensor):
        Q = self.linear_Q(Q_tensor)
        K = self.linear_K(K_tensor)
        V = self.linear_V(V_tensor)
        attention_matrix = QKVattention(Q, K, V, self.attention_QKV_dim)
        return attention_matrix

class transformer_block(nn.Module):
    def __init__(self, attention_QKV_dim) -> None:
        super().__init__()
        self.attn_layer = multihead_attention(attention_QKV_dim, attention_QKV_dim, attention_QKV_dim, attention_QKV_dim)
        self.MLP_layer = nn.Sequential(nn.Linear(attention_QKV_dim, attention_QKV_dim), nn.ReLU(), nn.Linear(attention_QKV_dim, attention_QKV_dim))
        
    def forward(self, Q_tensor, K_tensor, V_tensor):
        attention_matrix = self.attn_layer(Q_tensor, K_tensor, V_tensor)
        after_mlp = self.MLP_layer(attention_matrix)
        return after_mlp
        
class GenerativePretrainedTransformer(nn.Module):
    def __init__(self, Q_tensor_dim, K_tensor_dim, V_tensor_dim, attention_QKV_dim, transformer_layer_number):
        super().__init__()
        self.initial_attention = multihead_attention(Q_tensor_dim, K_tensor_dim, V_tensor_dim, attention_QKV_dim)
        self.multi_layer_transformer_blocks = nn.ModuleList([transformer_block(attention_QKV_dim) for i in range(transformer_layer_number)])
        
    def forward(self, Q_tensor, K_tensor, V_tensor):
        initial_attn = self.initial_attention(Q_tensor, K_tensor, V_tensor)
        for idx, block in enumerate(self.multi_layer_transformer_blocks):
            if idx == 0:
                out = block(initial_attn, initial_attn, initial_attn)
            else:
                out = block(out, out, out)
        return out


########## Prepare for dataset ##########
import random
import string

# Make dataset
def generate_random_string(length):
    characters = string.ascii_uppercase + string.digits + '",.'
    return ''.join(random.choice(characters) for _ in range(length))

# length = 20
# random_string = generate_random_string(length)  # Generate a random string of length 20
# print(random_string)

# Random databased dataset

########## Train ##########

gpt = GenerativePretrainedTransformer(5, 5, 5, 3, 10)
print(gpt)

Q_tensor = torch.rand(4, 5)
K_tensor = torch.rand(4, 5)
V_tensor = torch.rand(4, 5)
print(Q_tensor)
out = gpt(Q_tensor, K_tensor, V_tensor)
print(out)

import torch
import torch.nn as nn
from torch.nn import functional as F
from config.model_config import model_config
from config.theological_knowledge_base import theological_knowledge_base

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        return self.out(context)

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            model_config["model_architecture"]["embedding_dim"],
            model_config["model_architecture"]["hidden_size"]
        )
        self.position_embedding = nn.Embedding(
            model_config["model_architecture"]["max_position_embeddings"],
            model_config["model_architecture"]["hidden_size"]
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_config["model_architecture"]["hidden_size"],
                nhead=model_config["model_architecture"]["num_attention_heads"],
                dim_feedforward=model_config["model_architecture"]["intermediate_size"],
                dropout=model_config["model_architecture"]["hidden_dropout_prob"]
            ),
            num_layers=model_config["model_architecture"]["num_hidden_layers"]
        )
        self.classifier = nn.Linear(
            model_config["model_architecture"]["hidden_size"],
            model_config["model_architecture"]["type_vocab_size"]
        )
        self.theological_attention = MultiHeadAttention(
            model_config["model_architecture"]["hidden_size"],
            model_config["model_architecture"]["num_attention_heads"]
        )
        self.theological_knowledge_base = theological_knowledge_base

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        positions = torch.arange(0, input_ids.size(1)).expand(input_ids.size(0), -1)
        position_embeddings = self.position_embedding(positions)
        embeddings = embeddings + position_embeddings

        # Apply theological attention
        theological_embeddings = self.theological_attention(embeddings)

        output = self.transformer(theological_embeddings)
        logits = self.classifier(output)
        return logits

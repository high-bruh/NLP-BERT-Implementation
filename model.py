import torch
import torch.nn as nn
import math


class BERTEmbeddings(nn.Module):
    """Construct embeddings from word, position and token_type embeddings."""
    
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
        # Register position_ids as buffer (not trained)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, hidden_size, num_attention_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        
    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size = hidden_states.size(0)
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        output = self.dense(context_layer)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states


class TransformerLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_attention_heads, dropout)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)
        self.output_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with residual connection
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        hidden_states = self.attention_norm(hidden_states + attention_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(hidden_states)
        ff_output = self.dropout(ff_output)
        hidden_states = self.output_norm(hidden_states + ff_output)
        
        return hidden_states


class BERTEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    
    def __init__(self, num_layers, hidden_size, num_attention_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class MiniBERT(nn.Module):
    """Mini BERT model for MLM and NSP pre-training."""
    
    def __init__(
        self,
        vocab_size,
        hidden_size=256,
        num_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512,
        type_vocab_size=2,
        dropout=0.1
    ):
        super().__init__()
        
        self.embeddings = BERTEmbeddings(
            vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout
        )
        
        self.encoder = BERTEncoder(
            num_layers, hidden_size, num_attention_heads, intermediate_size, dropout
        )
        
        # MLM head
        self.mlm_dense = nn.Linear(hidden_size, hidden_size)
        self.mlm_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.mlm_decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))
        self.mlm_decoder.bias = self.mlm_bias
        
        # NSP head
        self.nsp_dense = nn.Linear(hidden_size, hidden_size)
        self.nsp_classifier = nn.Linear(hidden_size, 2)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        
        # Convert attention mask to attention scores format
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Get embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        # Pass through encoder
        encoder_output = self.encoder(embedding_output, extended_attention_mask)
        
        # MLM predictions
        mlm_hidden = self.mlm_dense(encoder_output)
        mlm_hidden = nn.functional.gelu(mlm_hidden)
        mlm_hidden = self.mlm_norm(mlm_hidden)
        mlm_logits = self.mlm_decoder(mlm_hidden)
        
        # NSP predictions (using [CLS] token)
        cls_output = encoder_output[:, 0]
        nsp_hidden = self.nsp_dense(cls_output)
        nsp_hidden = torch.tanh(nsp_hidden)
        nsp_logits = self.nsp_classifier(nsp_hidden)
        
        return mlm_logits, nsp_logits
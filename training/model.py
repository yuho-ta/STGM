"""
Audio classification model definition
Cross Attention Transformer
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionEncoderLayer(nn.Module):
    """
    Cross-attention Transformer encoder layer
    
    Perform cross-attention between two different modalities (audio and text),
    so that one modality can extract and integrate relevant information from the other.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU()):
        super(CrossAttentionEncoderLayer, self).__init__()
        
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Multi-head cross-attention layer
        # Using query=src, key=memory, value=memory makes src attend to memory
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Position-wise feed-forward network (FFN)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, memory, src_mask=None, src_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass of the cross-attention layer
        
        Args:
            src (torch.Tensor): Query tensor. Shape `(B, T_src, d_model)` (e.g., audio features)
            memory (torch.Tensor): Key/value tensor. Shape `(B, T_mem, d_model)` (e.g., text features)
            src_mask (torch.Tensor, optional): Query attention mask
            src_key_padding_mask (torch.BoolTensor, optional): Query key padding mask
                                                               `True` indicates positions to ignore
            memory_key_padding_mask (torch.BoolTensor, optional): Key padding mask for the memory (key/value)
                                                                  `True` indicates positions to ignore
        Returns:
            torch.Tensor: Output tensor of `src` after cross-attention and FFN. Shape is `(B, T_src, d_model)`
        """
        # Pre-normalization: apply normalization before attention
        src_norm = self.norm1(src)
        memory_norm = self.norm1(memory)
        
        # Cross-attention: src (query) attends to memory (key/value)
        # key_padding_mask masks the key side (memory side)
        attn_output, _ = self.cross_attention(
            query=src_norm,
            key=memory_norm,
            value=memory_norm,
            attn_mask=src_mask,
            key_padding_mask=memory_key_padding_mask
        )
        
        # Residual connection and dropout
        src = src + self.dropout1(attn_output)
        
        # Post-normalization and feed-forward network
        src_norm = self.norm2(src)
        src = src + self.dropout2(self.feedforward(src_norm))
        
        return src


class AttnPooling(nn.Module):
    """Attention pooling"""
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) - positions to ignore where True
        """
        attn_weights = self.attention(x)  # (batch, seq_len, 1)
        
        if mask is not None:
            # Set attention weights at masked positions to -inf
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1), float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=1)  # (batch, seq_len, 1)
        # Prevent NaNs when all positions are masked
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        pooled = (x * attn_weights).sum(dim=1)  # (batch, d_model)
        
        return pooled


class HierarchicalAttnPooling(nn.Module):
    """Hierarchical attention pooling (intra-segment + inter-segment)."""
    def __init__(self, d_model, segment_size=20):
        super().__init__()
        self.segment_size = segment_size
        self.intra_attn = AttnPooling(d_model)
        self.inter_attn = AttnPooling(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) - positions to ignore where True
        """
        batch_size, seq_len, d_model = x.size()
        num_segments = seq_len // self.segment_size

        if num_segments == 0:
            if mask is not None:
                return self.inter_attn(x, mask=mask)
            return x.mean(dim=1)

        # Split into segments: (batch, num_segments, segment_size, d_model)
        usable_len = num_segments * self.segment_size
        x_segments = x[:, :usable_len, :].reshape(
            batch_size, num_segments, self.segment_size, d_model
        )

        if mask is not None:
            mask_segments = mask[:, :usable_len].reshape(
                batch_size, num_segments, self.segment_size
            )
        else:
            mask_segments = None

        # Intra-segment attention pooling
        x_flat = x_segments.reshape(-1, self.segment_size, d_model)
        mask_flat = (
            mask_segments.reshape(-1, self.segment_size)
            if mask_segments is not None
            else None
        )
        segment_embeddings = self.intra_attn(x_flat, mask=mask_flat)
        segment_embeddings = segment_embeddings.view(batch_size, num_segments, d_model)

        # Inter-segment attention pooling
        if mask_segments is not None:
            segment_mask = mask_segments.all(dim=2)  # (batch, num_segments)
        else:
            segment_mask = None

        pooled = self.inter_attn(segment_embeddings, mask=segment_mask)
        return pooled


class CrossAttentionTransformer(nn.Module):
    """
    Cross-attention Transformer classifier
    
    Fuse aligned audio and text features using cross-attention and classify.
    
    Modes:
    - "multimodal": Use both audio and text (cross-attention)
    - "audio": Use only audio features
    - "text": Use only text features
    """
    def __init__(
        self,
        audio_dim: int = 768,  # hidden_dim of wav2vec2-base
        text_dim: int = 768,   # BERT-base hidden_dim (auto-detected)
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 1,
        num_classes: int = 1,
        dropout: float = 0.3,
        dim_feedforward: int = 3072,
        pooling: str = "mean",  # "mean", "cls", "attn", "hierarchical"
        hidden_mlp_size: int = 256,
        mode: str = "multimodal",  # "audio", "text", "multimodal"
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.pooling = pooling
        self.mode = mode
        
        if mode == "multimodal":
            # Input projection layer (unify different dimensions)
            self.audio_proj = nn.Linear(audio_dim, d_model)
            self.text_proj = nn.Linear(text_dim, d_model)
            
            # Cross-attention layers only
            self.cross_attention_layers = nn.ModuleList([
                CrossAttentionEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                ) for _ in range(num_layers)
            ])
            
            # Layer normalization between Transformer layers
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(num_layers - 1)
            ])
            
            self.dropout = nn.Dropout(dropout)
            
        elif mode == "audio":
            # Audio only: projection + encoder layers
            self.audio_proj = nn.Linear(audio_dim, d_model)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=num_layers
            )
            self.dropout = nn.Dropout(dropout)
            
        elif mode == "text":
            # Text only: projection + encoder layers
            self.text_proj = nn.Linear(text_dim, d_model)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=num_layers
            )
            self.dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'audio', 'text', or 'multimodal'")
        
        # Set pooling strategy
        if pooling == "attn":
            self.attn_pooling = AttnPooling(d_model)
        elif pooling == "hierarchical":
            self.attn_pooling = HierarchicalAttnPooling(d_model, segment_size=20)
        else:
            self.attn_pooling = None
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden_mlp_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mlp_size, num_classes)
        )
        
    
    def forward(
        self,
        audio_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            audio_features: (batch, seq_len, audio_dim) - aligned audio features (required in audio/multimodal mode)
            text_features: (batch, seq_len, text_dim) - text features (required in text/multimodal mode)
            audio_lengths: (batch,) - number of valid audio tokens (required in audio/multimodal mode)
            text_lengths: (batch,) - number of valid text tokens (required in text/multimodal mode)
            audio_mask: (batch, seq_len) - audio padding mask (positions to ignore where True; optional)
            text_mask: (batch, seq_len) - text padding mask (positions to ignore where True; optional)
        """
        if self.mode == "multimodal":
            # Multimodal mode
            if audio_features is None or text_features is None:
                raise ValueError("Both audio_features and text_features are required for multimodal mode")
            if audio_lengths is None or text_lengths is None:
                raise ValueError("Both audio_lengths and text_lengths are required for multimodal mode")
            
            # Projection
            audio_proj = self.audio_proj(audio_features)  # (batch, seq_len, d_model)
            text_proj = self.text_proj(text_features)      # (batch, seq_len, d_model)
            
            # Create padding mask (derive from lengths if not provided)
            if audio_mask is None:
                audio_mask = torch.arange(audio_proj.shape[1], device=audio_proj.device) >= audio_lengths.unsqueeze(1)
            if text_mask is None:
                text_mask = torch.arange(text_proj.shape[1], device=text_proj.device) >= text_lengths.unsqueeze(1)
            
            # Cross-attention: audio attends to text
            src = audio_proj  # Use audio as query
            memory = text_proj  # Use text as key/value
            
            for i, layer in enumerate(self.cross_attention_layers):
                src = layer(
                    src=src,
                    memory=memory,
                    src_key_padding_mask=audio_mask,
                    memory_key_padding_mask=text_mask
                )
                if i < len(self.norm_layers):
                    src = self.norm_layers[i](src)
                    src = self.dropout(src)
            
            # Apply pooling strategy
            lengths = audio_lengths
            mask = audio_mask
            
        elif self.mode == "audio":
            # Audio-only mode
            if audio_features is None:
                raise ValueError("audio_features is required for audio mode")
            if audio_lengths is None:
                raise ValueError("audio_lengths is required for audio mode")
            
            # Projection
            src = self.audio_proj(audio_features)  # (batch, seq_len, d_model)
            
            # Create padding mask
            if audio_mask is None:
                audio_mask = torch.arange(src.shape[1], device=src.device) >= audio_lengths.unsqueeze(1)
            
            # Encoder
            src = self.encoder(src, src_key_padding_mask=audio_mask)
            src = self.dropout(src)
            
            # Apply pooling strategy
            lengths = audio_lengths
            mask = audio_mask
            
        elif self.mode == "text":
            # Text-only mode
            if text_features is None:
                raise ValueError("text_features is required for text mode")
            if text_lengths is None:
                raise ValueError("text_lengths is required for text mode")
            
            # Projection
            src = self.text_proj(text_features)  # (batch, seq_len, d_model)
            
            # Create padding mask
            if text_mask is None:
                text_mask = torch.arange(src.shape[1], device=src.device) >= text_lengths.unsqueeze(1)
            
            # Encoder
            src = self.encoder(src, src_key_padding_mask=text_mask)
            src = self.dropout(src)
            
            # Apply pooling strategy
            lengths = text_lengths
            mask = text_mask
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Apply pooling strategy (common processing)
        if self.pooling == "mean":
            # Masked mean pooling
            src = (src * (~mask).unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1).float()
        elif self.pooling == "cls":
            src = src[:, 0, :]  # Use the first token
        elif self.pooling in ["attn", "hierarchical"] and self.attn_pooling is not None:
            src = self.attn_pooling(src, mask=mask)
        else:
            # Default: masked mean pooling
            src = (src * (~mask).unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1).float()
        
        # Classification
        output = self.classifier(src)  # (batch, num_classes)
        
        return output


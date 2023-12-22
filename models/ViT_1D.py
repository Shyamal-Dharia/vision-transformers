import torch 
from torch import nn

class patch_embedding(nn.Module):
    """Patch Embedding layer for Vision Transformer (ViT).

    Args:
        in_channels (int): Number of input channels in the sequence. Defaults to 1.
        patch_size (int): Size of patches to convert from input sequence. Defaults to 5.
        embedding_dim (int): Size of embedding to turn the sequence into. Defaults to 768.

    Attributes:
        patch_size (int): Size of patches to convert from input sequence.
        patcher (nn.Conv1d): 1D convolution layer for turning the sequence into patches.

    """
    def __init__(self, in_channels: int = 1, patch_size: int = 5, embedding_dim: int = 768):
        super().__init__()

        self.patch_size = patch_size
        # 1D convolution layer for turning the sequence into patches.
        self.patcher = nn.Conv1d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

    def forward(self, x):
        """Forward pass of the patch embedding layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor after converting the sequence into patches.

        Raises:
            AssertionError: If the sequence length is not divisible by the patch size.

        """
        # Assertion to check that inputs are compatible with model initialization Args.
        sequence_length = x.shape[-1]
        assert sequence_length % self.patch_size == 0, f"Sequence length must be divisible by patch size, sequence length: {sequence_length}, patch size: {self.patch_size}"

        x_patched = self.patcher(x)  # Patchify the sequence into small patches.

        return x_patched.permute(0, 2, 1)

class multi_head_attention_block(nn.Module):
    """Multi-head Attention Block for Vision Transformer (ViT).

    Args:
        embedding_dim (int): Size of the input embedding vector. Defaults to 768.
        num_heads (int): Number of self-attention heads. Defaults to 12.
        attention_dropout (float): Dropout rate for attention regularization. Defaults to 0.

    Attributes:
        layer_norm (nn.LayerNorm): Layer normalization for input embeddings.
        multi_head_attention (nn.MultiheadAttention): Multi-head attention layer.

    """
    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 msa_dropout: float = 0):
        super().__init__()

        # Normalize the layer which has shape of embedding dim.
        self.layer_norm = nn.LayerNorm(normalized_shape = embedding_dim)

        # Create multi-head attention layer.
        self.multi_head_attention = nn.MultiheadAttention(embed_dim = embedding_dim,
                                                          num_heads = num_heads,
                                                          dropout = msa_dropout,
                                                          batch_first = True)  # Does our batch dimension come first?

    def forward(self, x):
        """Forward pass of the multi-head attention block.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Output tensor after multi-head attention with the same shape as the input.

        """
        x = self.layer_norm(x)

        attention_output, _ = self.multi_head_attention(query = x,
                                                        key = x,
                                                        value = x,
                                                        need_weights = False)
        return attention_output
    

class MLPBlock(nn.Module):
    """Multi-Layer Perceptron (MLP) Block for Vision Transformer (ViT).

    Args:
        embedding_dim (int): Size of the input embedding vector. Defaults to 786.
        mlp_hidden_units (int): Number of units in the hidden layer of the MLP. Defaults to 3072.
        mlp_dropout (float): Dropout rate for regularization. Defaults to 0.1.

    Attributes:
        layer_norm (nn.LayerNorm): Layer normalization for input embeddings.
        mlp (nn.Sequential): Multi-Layer Perceptron consisting of linear layers, GELU activation, and dropout.

    """
    def __init__(self,
                 embedding_dim: int = 786,
                 mlp_hidden_units: int = 3072,
                 mlp_dropout: float = 0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_hidden_units),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(in_features=mlp_hidden_units,
                      out_features=embedding_dim),
            nn.Dropout(p=mlp_dropout)
        )

    def forward(self, x):
        """Forward pass of the MLP block.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Output tensor after the MLP block with the same shape as the input.

        """
        return self.mlp(self.layer_norm(x))


class transformer_encoder_block(nn.Module):
    """Creates a Transformer Encoder block."""
    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 mlp_hidden_unit: int = 3072,
                 mlp_dropout: float = 0.1,
                 msa_dropout: float = 0):
        super().__init__()

        self.msa_block = multi_head_attention_block(embedding_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    msa_dropout=msa_dropout)

        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                 mlp_hidden_units=mlp_hidden_unit,
                                 mlp_dropout=mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x

        return x

class transformer_encoder_block(nn.Module):
    """Creates a Transformer Encoder block."""
    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 mlp_hidden_unit: int = 3072,
                 mlp_dropout: float = 0.1,
                 msa_dropout: float = 0):
        super().__init__()

        self.msa_block = multi_head_attention_block(embedding_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    msa_dropout=msa_dropout)

        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                 mlp_hidden_units=mlp_hidden_unit,
                                 mlp_dropout=mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x

        return x


class ViT_1D(nn.Module):
    """Vision Transformer 1D (ViT) model for sequence classification.

    Args:
        sequence_length (int): Length of the input sequence. Defaults to 310.
        in_channels (int): Number of input channels in the sequence. Defaults to 1.
        patch_size (int): Size of patches to convert from input sequence. Defaults to 5.
        num_transformer_layers (int): Number of transformer encoder layers. Defaults to 12.
        embedding_dim (int): Size of the embedding vectors. Defaults to 768.
        mlp_hidden_unit (int): Number of units in the hidden layer of the MLP block. Defaults to 3072.
        num_heads (int): Number of self-attention heads in the transformer encoder blocks. Defaults to 12.
        msa_dropout (float): Dropout rate for attention regularization in the transformer encoder blocks. Defaults to 0.
        mlp_dropout (float): Dropout rate for regularization in the MLP block. Defaults to 0.1.
        embedding_dropout (float): Dropout rate for regularization in the embedding layer. Defaults to 0.1.
        num_classes (int): Number of output classes. Defaults to 1000.

    Attributes:
        num_patches (int): Number of patches in the sequence.
        class_embedding (nn.Parameter): Learnable parameter for the class token.
        position_embedding (nn.Parameter): Learnable parameter for the positional embeddings.
        embedding_dropout (nn.Dropout): Dropout layer for the embedding layer.
        patch_embedding (patch_embedding): Patch Embedding layer.
        transformer_encoder (nn.Sequential): Sequential container of transformer encoder blocks.
        classifier (nn.Sequential): Sequential container for the final classification layer.

    """
    def __init__(self,
                 sequence_length: int = 310,
                 in_channels: int = 1,
                 patch_size: int = 5,
                 num_transformer_layers: int = 12,
                 embedding_dim: int = 768,
                 mlp_hidden_unit: int = 3072,
                 num_heads: int = 12,
                 msa_dropout: float = 0,
                 mlp_dropout: float = 0.1,
                 embedding_dropout: float = 0.1,
                 num_classes: int = 1000):
        super().__init__()

        # Assertion to check that inputs are compatible with model initialization Args.
        assert sequence_length % patch_size == 0, f"Input size must be divisible by patch size, sequence length: {sequence_length}, patch size: {patch_size}"

        # Calculate the number of patches.
        self.num_patches = sequence_length // patch_size

        # Learnable parameter for the class token.
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        # Learnable parameter for the positional embeddings.
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches + 1, embedding_dim),
                                               requires_grad=True)

        # Create embedding dropout.
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # Initialize the patch embedding layer.
        self.patch_embedding = patch_embedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        # Initialize transformer encoder layers.
        self.transformer_encoder = nn.Sequential(
            *[transformer_encoder_block(embedding_dim=embedding_dim,
                                        num_heads=num_heads,
                                        mlp_hidden_unit=mlp_hidden_unit,
                                        mlp_dropout=mlp_dropout,
                                        msa_dropout=msa_dropout) for _ in range(num_transformer_layers)]
        )

        # Initialize classifier layer.
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):
        """Forward pass of the Vision Transformer 1D (ViT) model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor after the ViT model with shape (batch_size, num_classes).

        """
        batch_size = x.shape[0]

        # Expand the class token for each sequence in the batch.
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        # Apply patch embedding.
        x = self.patch_embedding(x)

        # Concatenate the class token with the patches.
        x = torch.cat((class_token, x), dim=1)

        # Add positional embeddings.
        x = self.position_embedding + x

        # Apply embedding dropout.
        x = self.embedding_dropout(x)

        # Apply transformer encoder layers.
        x = self.transformer_encoder(x)

        # Apply classifier layer.
        x = self.classifier(x[:, 0])

        return x



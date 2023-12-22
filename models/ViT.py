import torch
from torch import nn
import torchvision.models

class patch_embedding(nn.Module):
    """Patch Embedding layer for Vision Transformer (ViT).

    Args:
        in_channels (int): Number of color channels available for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert from input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn the image into. Defaults to 768.

    Attributes:
        patch_size (int): Size of patches to convert from input image into.
        patcher (nn.Conv2d): 2D convolution layer for turning the image into patches.
        flatten (nn.Flatten): Flatten the patches into a 1D sequence.

    """
    # Initialize the class with Args.
    def __init__(self, in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_dim: int = 768):
        super().__init__()

        self.patch_size = patch_size
        # 2D convolution layer for turning the image into patches.
        self.patcher = nn.Conv2d(in_channels = in_channels,
                                 out_channels = embedding_dim,
                                 kernel_size = patch_size,
                                 stride = patch_size,
                                 padding = 0)

        # Flatten the patches into a 1D sequence.
        self.flatten = nn.Flatten(start_dim = 2,  # Only flatten the feature map (16x16) dimensions into a single vector.
                                  end_dim = 3)

    def forward(self, x):
        """Forward pass of the patch embedding layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after converting the image into a 1D sequence.

        Raises:
            AssertionError: If the input size is not divisible by the patch size.

        """
        # Assertion to check that inputs are compatible with model initialization Args.
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input size must be divisible by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        x_patched = self.patcher(x)  # Patchify the image into small patches.
        x_flatten = self.flatten(x_patched)  # Convert patches into a 1D sequence learnable embedding vector.

        return x_flatten.permute(0, 2, 1)


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
                 attention_dropout: float = 0):
        super().__init__()

        # Normalize the layer which has shape of embedding dim.
        self.layer_norm = nn.LayerNorm(normalized_shape = embedding_dim)

        # Create multi-head attention layer.
        self.multi_head_attention = nn.MultiheadAttention(embed_dim = embedding_dim,
                                                          num_heads = num_heads,
                                                          dropout = attention_dropout,
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


class MLP_block(nn.Module):
    """Multi-Layer Perceptron (MLP) Block for Vision Transformer (ViT).

    Args:
        embedding_dim (int): Size of the input embedding vector. Defaults to 786.
        mlp_hidden_units (int): Number of units in the hidden layer of the MLP. Defaults to 3072.
        dropout (float): Dropout rate for regularization. Defaults to 0.1.

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
            nn.Linear(in_features = embedding_dim,
                      out_features = mlp_hidden_units),
            nn.GELU(),
            nn.Dropout(p = mlp_dropout),
            nn.Linear(in_features = mlp_hidden_units,
                      out_features = embedding_dim),
            nn.Dropout(p = mlp_dropout)
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
    """Transformer Encoder Block for Vision Transformer (ViT).

    Args:
        embedding_dim (int): Size of the input embedding vector. Defaults to 768.
        num_heads (int): Number of self-attention heads for the multi-head attention block. Defaults to 12.
        mlp_hidden_unit (int): Number of units in the hidden layer of the MLP block. Defaults to 3072.
        msa_dropout (float): Dropout rate for attention regularization in the multi-head attention block. Defaults to 0.
        mlp_dropout (float): Dropout rate for regularization in the MLP block. Defaults to 0.1.

    Attributes:
        msa_block (multi_head_attention_block): Multi-head Attention Block.
        mlp_block (MLP_block): Multi-Layer Perceptron (MLP) Block.

    """
    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 mlp_hidden_unit: int = 3072,
                 msa_dropout: float = 0,
                 mlp_dropout: float = 0.1):
        super().__init__()

        self.msa_block = multi_head_attention_block(embedding_dim = embedding_dim,
                                                    num_heads = num_heads,
                                                    attention_dropout = msa_dropout)

        self.mlp_block = MLP_block(embedding_dim = embedding_dim,
                                   mlp_hidden_units = mlp_hidden_unit,
                                   mlp_dropout = mlp_dropout)

    def forward(self, x):
        """Forward pass of the Transformer Encoder Block.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Output tensor after the Transformer Encoder Block with the same shape as the input.

        """
        # Apply the multi-head self-attention block with a residual connection.
        x = self.msa_block(x) + x

        # Apply the MLP block with a residual connection.
        x = self.mlp_block(x) + x

        return x

class ViT(nn.Module):
    """Vision Transformer (ViT) model for image classification.

    Args:
        img_size (int): Size of the input image. Defaults to 224.
        in_channels (int): Number of color channels available for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert from input image into. Defaults to 16.
        num_transformer_layers (int): Number of transformer encoder layers. Defaults to 12.
        embedding_dim (int): Size of the embedding vectors. Defaults to 768.
        mlp_hidden_unit (int): Number of units in the hidden layer of the MLP block. Defaults to 3072.
        num_heads (int): Number of self-attention heads in the transformer encoder blocks. Defaults to 12.
        msa_dropout (float): Dropout rate for attention regularization in the transformer encoder blocks. Defaults to 0.
        mlp_dropout (float): Dropout rate for regularization in the MLP block. Defaults to 0.1.
        embedding_dropout (float): Dropout rate for regularization in the embedding layer. Defaults to 0.1.
        num_classes (int): Number of output classes. Defaults to 1000.
        weights (bool): Whether to load weights. Defaults to False.
        model_name (str): Name of the model for loading weights. Defaults to None.

    Attributes:
        num_patches (int): Number of patches in the image.
        class_embedding (nn.Parameter): Learnable parameter for the class token.
        position_embedding (nn.Parameter): Learnable parameter for the positional embeddings.
        embedding_dropout (nn.Dropout): Dropout layer for the embedding layer.
        patch_embedding (patch_embedding): Patch Embedding layer.
        transformer_encoder (nn.Sequential): Sequential container of transformer encoder blocks.
        classifier (nn.Sequential): Sequential container for the final classification layer.

    """
    def __init__(self,
                 img_size: int = 224,
                 in_channels: int = 3,
                 patch_size: int = 16,
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
        assert img_size % patch_size == 0, f"Input size must be divisible by patch size, image shape: {img_size}, patch size: {patch_size}"

        # Calculate the number of patches.
        self.num_patches = (img_size * img_size) // patch_size**2

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
        """Forward pass of the Vision Transformer (ViT) model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, img_size, img_size).

        Returns:
            torch.Tensor: Output tensor after the ViT model with shape (batch_size, num_classes).

        """
        batch_size = x.shape[0]

        # Expand the class token for each image in the batch.
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








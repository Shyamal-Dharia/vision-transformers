
import torch
import torchvision
from torchvision import datasets
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
import argparse

class PatchEmbedding(nn.Module):
    """
    Patch Embedding module for Vision Transformer.

    Args:
        in_channels (int): Number of input channels (default: 3).
        patch_size (int): Size of each patch (default: 16).
        embedding_dim (int): Dimension of the embedded representation (default: 768).
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768):
        super().__init__()

        self.patch_size = patch_size
        self.patcher = nn.Conv2d(
            in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0,
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        """
        Forward pass of the Patch Embedding module.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Flattened and permuted representation of patches.
        """
        image_resolution = x.shape[-1]
        assert (
            image_resolution % self.patch_size == 0
        ), f"Input size must be divisible by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        x_patched = self.patcher(x)
        x_flatten = self.flatten(x_patched)

        return x_flatten.permute(0, 2, 1)

def load_model_with_trained_weights(data_folder: str):
    """
    Load a Vision Transformer model with pre-trained weights.

    Args:
        data_folder (str): Path to the data folder.

    Returns:
        torch.nn.Module: Loaded Vision Transformer model.
        list: List of class names.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)
    
    vit_transforms = pretrained_vit_weights.transforms()

    dataset = datasets.ImageFolder(root=data_folder, transform=vit_transforms)
    class_names = dataset.classes

    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    pretrained_vit.heads = torch.nn.Linear(in_features=768, out_features=len(class_names))
    pretrained_vit.load_state_dict(torch.load("ViT_model.pth"))
    pretrained_vit.to(device)

    return pretrained_vit, class_names

def img_prediction(model: torch.nn.Module, 
                   image_tensor: torch.Tensor):
    """
    Make a prediction using the provided image tensor and the model.

    Args:
        model (torch.nn.Module): Vision Transformer model.
        image_tensor (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: Model output.
    """
    output = model(image_tensor.unsqueeze(0)).to(device)
    pred = torch.argmax(output)
    print(f"Prediction from the model: {class_names[pred]}")
    return output

def attention_scores(model: torch.nn.Module, 
                   image_tensor: torch.Tensor):
    """
    Compute attention scores for the provided image tensor and model.

    Args:
        image_tensor (torch.Tensor): Input image tensor.
        model (torch.nn.Module): Vision Transformer model.

    Returns:
        torch.Tensor: Attention scores.
    """
    in_channels = 3
    patch_size = 16
    embedding_dim = 768
    batch_size = 1
    class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                   requires_grad=True).to(device)
    class_token = class_embedding.expand(batch_size, -1, -1).to(device)

    patch_embedding = PatchEmbedding(in_channels, patch_size, embedding_dim).to(device)
    out = patch_embedding(image_tensor.unsqueeze(0)).to(device)
    out = torch.cat((class_token, out), dim=1).to(device)

    attention_outputs = []

    for layer in model.encoder.layers:
        output, attention_weights = layer.self_attention(out, out, out)
        attention_outputs.append(attention_weights)

    attention_score = torch.cat(attention_outputs)
    return attention_score

def scores_to_viz(image_tensor, attention_score):
    """
    Convert attention scores to visualization.

    Args:
        image_tensor (torch.Tensor): Input image tensor.
        attention_score (torch.Tensor): Attention scores.

    Returns:
        numpy.ndarray: Attention visualizations.
    """
    patch_size = 16
    w, h = image_tensor.shape[1] - image_tensor.shape[1] % patch_size, image_tensor.shape[2] - \
        image_tensor.shape[2] % patch_size
    img = image_tensor[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attention_score = attention_score.unsqueeze(0)
    nh = attention_score.shape[1]
    attentions = attention_score[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="bilinear")[0].detach().cpu().numpy()

    return attentions

def plot_attention_and_save(img, attention):
    """
    Plot attention maps and save the figure.

    Args:
        img (numpy.ndarray): Original image.
        attention (numpy.ndarray): Attention maps.
        save_path (str): Path to save the figure.
    """
    n_heads = attention.shape[0]

    plt.figure(figsize=(16, 4))
    text = ["Original Image", "Head Mean"]
    for i, fig in enumerate([img, np.max(attention, 0)]):
        plt.subplot(1, 2, i+1)
        plt.imshow(fig, cmap='inferno')
        plt.title(text[i])
        plt.colorbar()

    plt.savefig("overview.png")
    plt.close()

    plt.figure(figsize=(10, 10))
    for i in range(n_heads):
        plt.subplot(n_heads//3, 3, i+1)
        plt.imshow(attention[i], cmap='inferno')
        plt.title(f"Head n: {i+1}")
        plt.colorbar()

    plt.tight_layout()
    plt.savefig("individual_heads.png")
    plt.close()

def process_image(image_path, img_size):
    """
    Process the input image for model input.

    Args:
        image_path (str): Path to the input image.
        img_size (int): Size of the input image.

    Returns:
        torch.Tensor: Processed image tensor.
    """
    img = Image.open(image_path)
    vit_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    image_tensor = vit_transforms(img).to(device)
    return image_tensor

def visualize_attention_score(model, image_tensor):
    """
    Visualize attention scores for the provided image tensor and model.

    Args:
        model (torch.nn.Module): Vision Transformer model.
        image_tensor (torch.Tensor): Input image tensor.
    """
    attention_score = attention_scores(model, image_tensor)
    attentions = scores_to_viz(image_tensor, attention_score)
    img = image_tensor.detach().cpu()
    img = img.permute(1, 2, 0)
    plot_attention_and_save(img, attentions)

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize attention scores using Vision Transformer.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--img_size", type=int, default=224, help="Size of the input image.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    image_path = "data/caltech101/101_ObjectCategories/stop_sign/image_0023.jpg"
    img_size = 224

    pretrained_model, class_names = load_model_with_trained_weights(data_folder="data/caltech101/101_ObjectCategories/")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_tensor = process_image(image_path = args.image_path,
                                img_size = args.img_size)
    
    img_prediction(model=pretrained_model,
                   image_tensor=img_tensor)
    
    visualize_attention_score(model=pretrained_model, 
                              image_tensor=img_tensor)

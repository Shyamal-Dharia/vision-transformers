from models import ViT, ViT_1D
import torch

def example_image_data():
    """
    Example for image data using ViT model.
    """
    # Generate random image data with shape (batch_size, channels, height, width)
    image_data = torch.randn(1, 3, 224, 224)

    # Instantiate the ViT model for image data
    vit_model = ViT(num_classes=5)

    # Forward pass to get the model output
    output = vit_model(image_data)

    # Print the output
    print("Image Data Example:")
    print(output)

def example_time_series_data():
    """
    Example for time-series data using ViT_1D model.
    """
    # Generate random time-series data with shape (batch_size, channels, sequence_length)
    time_series_data = torch.randn(1, 1, 500)

    # Instantiate the ViT_1D model for time-series data
    vit_1d_model = ViT_1D(num_classes=5)

    # Forward pass to get the model output
    output = vit_1d_model(time_series_data)

    # Print the output
    print("\nTime-Series Data Example:")
    print(output)

if __name__ == "__main__":
    # Run the examples
    example_image_data()
    example_time_series_data()

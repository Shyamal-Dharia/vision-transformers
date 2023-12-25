
import os
from modules import engine
import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import argparse

def parse_args():
    """
    Parse command-line arguments for training ViT model on datasets.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train ViT model on Caltech101 dataset")
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--data_folder', type=str, default='data/caltech101/101_ObjectCategories/', 
                        help='Path to the folder where data is downloaded')
    return parser.parse_args()


args = parse_args()

NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
data_folder = args.data_folder


device = "cuda" if torch.cuda.is_available() else "cpu"

# get the pretrained weights.
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

#get the base ViT model.
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)

#get the transforms of pretrained ViT weights, so that we can transform our caltech101 data accordingly.
vit_transforms = pretrained_vit_weights.transforms()

dataset = datasets.ImageFolder(root=data_folder, transform=vit_transforms)

#get class_names in a list
class_names = dataset.classes

# Split the dataset into training and testing sets
train_size = int(0.80 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader for training and testing
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



#Freeze the top layers
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

#recreate the classifier layer, according to the number of classes in the caltech data.
pretrained_vit.heads = torch.nn.Linear(in_features=768, out_features=len(class_names))


# Create optimizer and loss function
optimizer = torch.optim.Adam(params=pretrained_vit.parameters(),
                             lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the classifier head of the pretrained ViT feature extractor model
pretrained_vit_results = engine.train(model=pretrained_vit,
                                      train_dataloader=train_loader,
                                      test_dataloader=test_loader,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=NUM_EPOCHS,
                                      device=device)
# save the model 
torch.save(obj = pretrained_vit.state_dict(), f = "ViT_model.pth")
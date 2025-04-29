import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG19_Weights
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

#Standardizing the content and style image based off a gaussian distribution
global_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
global_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def get_features_per_layer(image, model):
    #This function extracts the features of the iamge per layer of the VGG19 model
    layers = {
        '0': 'conv1_1', 
        '5': 'conv2_1', 
        '10': 'conv3_1',
        '19': 'conv4_1', 
        '21': 'content'
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def gram_matrix(tensor):
    #finds the gram matrix at a given layer which is the dot product of the feature map
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t()) / (d * h * w)

def style_loss(target_features, style_features):
    #calculates the style loss by extracting the gram matrix of the style and target features at each given layer and find the mse 
    #between the two
    loss = 0
    for layer in style_features:
        target_gram = gram_matrix(target_features[layer])
        style_gram = gram_matrix(style_features[layer])
        loss += F.mse_loss(target_gram, style_gram)
    return loss


def run_model(vgg, content_image, style_image, name):

    content_features = get_features_per_layer(content_image, vgg)
    style_features = get_features_per_layer(style_image, vgg)


    target = content_image.clone().requires_grad_(True).to(device)

    # ------HYPERPARAMETERS------
    epochs = 1000 #High number of epochs
    learning_rate = 0.1 #Higher learning rate to increase speed of convergence
    style_weight = 1e10  #Heavily weighting the style loss
    content_weight = 1  #Low content loss weighting since we are initializing from the content image
    
    #Learning rate scheduler
    step_size = 250
    gamma = 0.6

    #Using ADAM as the standard optimizer
    optimizer = optim.Adam([target], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for _ in range(epochs):
        optimizer.zero_grad()
        
        target_features = get_features_per_layer(target, vgg)
        s_loss = style_loss(target_features, style_features)
        c_loss = F.mse_loss(target_features['content'], content_features['content'])
        
        #Calculate total loss
        total_loss = style_weight * s_loss + content_weight * c_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

    # Denormalizing the image for saving
    generated_image = (target.cpu().clone() * global_std + global_mean).clamp(0, 1)
    
    #Final transformation into a PIL image
    generated_pil = transforms.ToPILImage()(generated_image)
    
    generated_pil.save(f"../output/{name}.png")
    print("Styled image saved as stylized_output.png")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Using the VGG19 model weights for feature extraction and transfer
    vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad_(False)

    # Using AvgPool2d instead of MaxPool2d. MaxPooling tends to pull the most dominant features
    # from the image, while AvgPooling gives a more generalized representation
    for i, layer in enumerate(vgg):
        if isinstance(layer, torch.nn.MaxPool2d):
            vgg[i] = torch.nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)

    style_dir = "../style"
    content_dir = "../content"

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])


    if not os.path.exists("../output"):
        os.makedirs("../output")
    for style in os.listdir(style_dir):
        for content in os.listdir(content_dir):
            print(f"Processing {style} with {content}")
            content_image = transform(Image.open(f"{content_dir}/{content}")).unsqueeze(0).to(device)
            style_image = transform(Image.open(f"{style_dir}/{style}")).unsqueeze(0).to(device) 
            run_model(vgg, content_image, style_image, name = f"{style}_{content}")
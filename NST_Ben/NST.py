import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG19_Weights
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# standardize input images. 
transform = transforms.Compose([
    transforms.Resize((680, 680)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #TODO play around with normalizaiton factor?
])

#load images and convert them to the same colorset
def load_image(image_path, to_grayscale=False):
    """Loads an image and optionally converts it to grayscale."""
    image = Image.open(image_path)
    if to_grayscale:
        image = image.convert("L").convert("RGB")
    else:
        image = image.convert("RGB")
    image = transform(image).unsqueeze(0)
    return image.to(device)

content_image = load_image("yorkshire-terrier-sitting-on-decking.jpg", to_grayscale=False)
style_image = load_image("Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg.webp") #Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg.webp
#output_panels/panel_0.png

# Using the VGG19 model weights for feature extraction and transfer
vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad_(False)


def extract_features(image, model, layers=None):
    """Extracts features from specified layers of the model."""
    if layers is None:
        layers = {
            '0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
            '19': 'conv4_1', '21': 'content'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def gram_matrix(tensor):
    """Computes the Gram matrix for style loss."""
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t()) / (d * h * w)

def style_loss(target_features, style_features):
    """Computes style loss."""
    loss = 0
    for layer in style_features:
        target_gram = gram_matrix(target_features[layer])
        style_gram = gram_matrix(style_features[layer])
        loss += F.mse_loss(target_gram, style_gram)
    return loss

def content_loss(target_features, content_features):
    """Computes content loss."""
    return F.mse_loss(target_features['content'], content_features['content'])


content_features = extract_features(content_image, vgg)
style_features = extract_features(style_image, vgg)


target = content_image.clone().requires_grad_(True).to(device)


epochs = 1000
learning_rate = 0.1
style_weight = 1e10  #Weighting for style loss
content_weight = 1  
step_size = 250
gamma = 0.6

optimizer = optim.Adam([target], lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

for epoch in range(epochs):
    optimizer.zero_grad()
    
    target_features = extract_features(target, vgg)
    s_loss = style_loss(target_features, style_features)
    c_loss = content_loss(target_features, content_features)
    

    total_loss = style_weight * s_loss + content_weight * c_loss
    total_loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Style Loss: {s_loss.item():.6f}, Content Loss: {c_loss.item():.6f}, Total Loss: {total_loss.item():.6f}")

def denormalize_image(tensor):

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor.cpu().clone() * std + mean  
    return tensor.clamp(0, 1)

def show_image(tensor_image, title="Styled Image"):
    
    image = tensor_image.cpu().clone().detach()  
    image = image.squeeze(0)  
    image = transforms.ToPILImage()(image)

    plt.imshow(image)
    plt.title(title)
    plt.axis("off")  
    plt.show()


styled_image = denormalize_image(target.cpu().squeeze(0)) 
show_image(styled_image)

styled_pil = transforms.ToPILImage()(styled_image)
styled_pil.save("stylized_output.png")
print("Styled image saved as stylized_output.png")

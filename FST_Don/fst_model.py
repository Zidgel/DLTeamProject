import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from transform_network import TransformNet


class fst_model(nn.Module):
    def __init__(self, style_image, device):
        super().__init__()
        self.device = device
        self.vgg16 = models.vgg16().features  # no pretrained=True here
        self.vgg16.load_state_dict(torch.load("vgg16_features.pth"))
        self.vgg16.eval()
        self.vgg16 = self.vgg16.to(device)
        for param in self.vgg16.parameters():
            param.requires_grad = False

        self.net = TransformNet(self.device).to(self.device)

        self.activations = {}
        self.vgg16[3].register_forward_hook(self.get_activation("relu1_2"))
        self.vgg16[8].register_forward_hook(self.get_activation("relu2_2"))
        self.vgg16[15].register_forward_hook(self.get_activation("relu3_3"))
        self.vgg16[22].register_forward_hook(self.get_activation("relu4_3"))
        self.style_image = style_image
        self.style_features = self.get_features(self.style_image)

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output

        return hook

    def get_features(self, image, is_detach=False):
        _ = self.vgg16.forward(image)
        if is_detach:
            return {
                "relu1_2": self.activations["relu1_2"].clone().detach(),
                "relu2_2": self.activations["relu2_2"].clone().detach(),
                "relu3_3": self.activations["relu3_3"].clone().detach(),
                "relu4_3": self.activations["relu4_3"].clone().detach(),
            }
        return {
            "relu1_2": self.activations["relu1_2"],  # .clone().detach(),
            "relu2_2": self.activations["relu2_2"],  # .clone().detach(),
            "relu3_3": self.activations["relu3_3"],  # .clone().detach(),
            "relu4_3": self.activations["relu4_3"],  # .clone().detach(),
        }

    def forward(self, x, style_image):
        out = self.net(x)
        # with torch.no_grad():
        #     _ = self.vgg16(x)
        # input_act_1 = self.activations["relu1_2"]
        # input_act_2 = self.activations["relu2_2"]
        # input_act_3 = self.activations["relu3_3"]
        # input_act_4 = self.activations["relu4_4"]
        # with torch.no_grad():
        #     _ = self.vgg16(style_image)
        # style_act_1 = self.activations["relu1_2"]
        # style_act_2 = self.activations["relu2_2"]
        # style_act_3 = self.activations["relu3_3"]
        # style_act_4 = self.activations["relu4_4"]
        # with torch.no_grad():
        #     _ = self.vgg16(style_image)
        # content_act = self.activations["relu3_3"]
        return out

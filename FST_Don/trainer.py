from fst_model import fst_model
import torch.nn as nn
import torch
import torch.optim as optim


class trainer:
    def __init__(
        self,
        style_image: torch.Tensor,
        epochs,
        content_loader,
        device,
        lr,
        style_beta,
    ):
        self.model = fst_model(style_image.to(device), device).to(device)
        self.epochs = epochs
        self.content_loader = content_loader
        self.device = device
        self.style_image = style_image.to(device)
        self.lr = lr
        self.style_beta = style_beta

    def train(self):
        optimizer = optim.Adam(self.model.net.parameters(), lr=self.lr)
        mse_loss = nn.MSELoss()
        tv_loss = 0

        for epoch in range(self.epochs):
            for i, (content_img, _) in enumerate(self.content_loader):
                if i == 50:
                    break
                content_img = content_img.to(self.device)
                y_hat = self.model.net.forward(content_img).to(self.device)
                y_content = content_img

                y_hat_features = self.model.get_features(y_hat)
                y_content_features = self.model.get_features(y_content)
                content_loss = mse_loss.forward(
                    y_hat_features["relu3_3"], y_content_features["relu3_3"]
                )

                # --- Style Loss ---
                style_loss = 0
                for layer in ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]:
                    out_gram = gram_matrix(y_hat_features[layer])
                    style_gram = gram_matrix(self.model.style_features[layer])
                    style_loss += mse_loss(out_gram, style_gram)

                total_loss = content_loss + self.style_beta * style_loss

                # --- Backward ---
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # --- Logging ---
                if i % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}], Batch [{i+1}], "
                        f"Content: {content_loss.item():.4f}, "
                        f"Style: {self.style_beta * style_loss.item():.4f}, "
                        f"Total: {total_loss.item():.4f}"
                    )


def gram_matrix(features):
    if len(features.shape) == 4:
        B, C, H, W = features.size()
    else:
        C, H, W = features.size()
        B = 1
    features = features.view(B, C, H * W)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (C * H * W)

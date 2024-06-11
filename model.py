import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    device, dtype = None, None
    if isinstance(sigma, torch.Tensor):
        device, dtype = sigma.device, sigma.dtype
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
    return gauss / gauss.sum()

center_bias_density = np.load('center_bias_density.npy')
log_center_bias_density = np.log(center_bias_density)

H, W = 224, 224 
log_center_bias_density_resized = np.resize(log_center_bias_density, (H, W))
log_center_bias_density_tensor = torch.tensor(log_center_bias_density_resized, dtype=torch.float32)
log_center_bias_density_tensor = log_center_bias_density_tensor.unsqueeze(0).unsqueeze(0)
log_center_bias_density_tensor = nn.Parameter(log_center_bias_density_tensor, requires_grad=False)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = models.vgg16_bn(pretrained=False).features
    
    def forward(self, x):
        x = self.model(x)
        return x

class Decoder(nn.Module):
    def __init__(self, window_size=25, sigma=11.2):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1)

        # upsampling
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        kernel = gaussian(window_size, sigma)
        self.kernel = nn.Parameter(kernel.view(1, 1, window_size, 1), requires_grad=False)

        self.log_center_bias_density = log_center_bias_density_tensor

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        x = self.upsample(x) 

        x = F.conv2d(x, self.kernel, padding='same')
        x += self.log_center_bias_density.to(x.device)

        return x

class EyeFixationNetwork(nn.Module):
    def __init__(self):
        super(EyeFixationNetwork, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    model = EyeFixationNetwork()
    input_data = torch.randn(1, 3, 224, 224)
    output = model(input_data)
    print("Output shape:", output.shape)  # Output shape should be (batch_size, 1, 224, 224)

import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV2Cus(nn.Module):
    def __init__(self, n_sections):
        super(MobileNetV2Cus, self).__init__()
        self.mobilenetv2 = models.mobilenet_v2(pretrained=False, progress=True, num_classes=n_sections)
        self.conv = nn.Sequential(
            nn.Conv2d(1280, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, n_sections)
        )

    def forward(self, x):
        x = self.mobilenetv2.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def emb(self, x):
        x = self.mobilenetv2.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return x
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGG16(nn.Module):
    def __init__(self, n_classes=10):
        super(VGG16, self).__init__()
        self.n_classes = n_classes

        backbone = models.vgg16_bn(pretrained=True)
        self.encoder = nn.Sequential(*(list(backbone.children())[:-2]))

        self.cls = nn.Linear(512, self.n_classes)

    def forward(self, x, return_embs=False):
        z = self.extract_emb(x)
        x = self.classify(z)

        if return_embs:
            return x, z
        return x

    def extract_emb(self, x):
        x = self.encoder(x)

        return x.view(x.size(0), -1)

    def classify(self, x):
        x = self.cls(x)

        return F.log_softmax(x, dim=1)

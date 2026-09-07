import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    
    def forward(self, feature_map):
        return F.adaptive_avg_pool2d(feature_map, 1).squeeze(-1).squeeze(-1)

class ImageClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        self.num_classes = num_classes

        feat_dim = 2048

        feature_extractor = models.resnet50(pretrained=True)
        feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-2])

        self.feature_extractor = feature_extractor
        self.avgpool = GlobalAvgPool2d()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.feature_extractor(x)
        pooled_feats = self.avgpool(feats)
        logits = self.fc(pooled_feats)


        return logits


class ImageClassifier_feats(torch.nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier_feats, self).__init__()
        self.num_classes = num_classes

        feat_dim = 2048

        feature_extractor = models.resnet50(pretrained=True)
        feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-2])

        self.feature_extractor = feature_extractor
        self.avgpool = GlobalAvgPool2d()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.feature_extractor(x)
        pooled_feats = self.avgpool(feats)
        logits = self.fc(pooled_feats)

        return logits, feats
import torch.nn as nn
import torch.nn.functional as F

class ENN(nn.Module):
    def __init__(self, classifier, activation='elu'):
        super().__init__()
        self.classifier = classifier
        self.activation = activation

    def forward(self, x):
        x = self.classifier(x)
        return F.elu(x) + 2

# The activation function is f(x) = ELU(x)+2. It is better than ReLU.
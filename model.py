import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import cv2

class Model(nn.Module):

    def __init__(self, num_bins=2):
        super().__init__()
        self.num_bins = num_bins

        # Build the CNN feature extractor
        self.backbone = torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        
        self.dropout = nn.Dropout(0.2)

        # Build a FC heads, taking both the image features and the intention as input
        self.fc_speed = nn.Sequential(
                    nn.Linear(576, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
        )
        
        self.fc_angle = nn.Sequential(
                    nn.Linear(576, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
        )
        
    def forward(self, image):
        # Map images to feature vectors
        feature = self.backbone(image)
        feature = self.dropout(feature)
        # Cast intention to one-hot encoding 
        # Predict control
        speed = self.fc_speed(feature)
        angle = self.fc_angle(feature)
        
        speed = F.sigmoid(speed)
        angle = F.tanh(angle)
        
        control = torch.cat([speed, angle], dim=1)
        # Return control as a categorical distribution
        return control
    
class Model2(nn.Module):
    def __init__(self, action_dim, max_action):
        super(Model2, self).__init__()

        # ONLY TRU IN CASE OF DUCKIETOWN:
        flat_size = 32 * 9 * 14

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.adap = nn.AdaptiveAvgPool2d((5, 5))

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(800, 512)
        self.lin2 = nn.Linear(512, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = self.bn1(self.lr(self.conv1(x)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = self.adap(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(x)
        x = self.lr(self.lin1(x))

        # because we don't want our duckie to go backwards
        x = self.lin2(x)
        x[:, 0] = self.sigm(x[:, 0])  # because we don't want the duckie to go backwards
        x[:, 1] = self.tanh(x[:, 1]) * 10

        return x
    
if __name__ == "__main__":
    model = Model()
    test = torch.rand(1, 3, 480, 640)
    out = model(test)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"No. of parameters: {num_params:,}")
    print(out.shape)
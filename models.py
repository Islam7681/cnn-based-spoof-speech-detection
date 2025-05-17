import torch.nn as nn

# this is a simple cnn that takes in audio features (usually 2d like log-mel)
# basically, it runs the input through a few layers of convolution, then flattens and classifies
class CNNAudioNet(nn.Module):
    def __init__(self, input_channels=1, n_classes=1):
        super(CNNAudioNet, self).__init__()
        # three convolution blocks, each with conv, relu, batchnorm, then pooling
        # these will reduce spatial size and increase channel depth
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            
            # second block: features go deeper
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            # third block: a bit more channel depth before global pool
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2,2))      # pool over both axes to shrink further
        )
        # this just averages over everything left in time/freq (gives us a summarized vector for each example)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        # this part maps our pooled features (size 64) into a prediction
        self.classifier = nn.Sequential(
            nn.Flatten(),    # collapse to [batch, 64]
            nn.Linear(64, 32),  # a small fc to 32 (just some compression mixing)
            nn.ReLU(),
            nn.Dropout(0.3),  # helps prevent overfitting a bit, but we still were getting it after epoch 9
            nn.Linear(32, n_classes) # goes to just 1 output, for binary
        )
    def forward(self, x):
        # forward pass for input x
        x = self.conv_block(x)
        x = self.global_pool(x) # now reduced to [batch, 64, 1, 1]
        x = self.classifier(x)  # classifier returns [batch, 1]
        return x.squeeze(-1)    # just take off last dim (if present)
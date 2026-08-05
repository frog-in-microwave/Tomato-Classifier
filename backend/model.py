import torch
import torch.nn as nn

class Tomato_classifier(nn.Module):
  def __init__(self):
    super().__init__()


    # creating convolutional layers with nn.Conv2d class and a pool with nn.MaxPool2d class

    # 3 x 224 x 224 in, 32 x 112 x 112 out.
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
    # 32 x 112 x 112 in, 64 x 56 x 56 out.
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    # 64 x 56 x 56 in, 128 x 28 x 28 out.
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    # 128 x 28 x 28 in, 128 x 14 x 14
    self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    # 128 x 14 x 14 in, 128 x 7 x 7
    self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    # n x m x m in, n x m/2 x m/2 out
    self.pool = nn.MaxPool2d(2, 2)


    # creating linear layers

    # 128 x 28 x 28 = 6272 so 6272 inputs an 6 outputs, one for each category
    self.fc1 = nn.Linear(6272, 512)
    self.fc2 = nn.Linear(512, 128)
    self.fc3 = nn.Linear(128, 6)

  def forward(self, x):

      # Convolutional feature extractor
      x = self.pool(torch.relu(self.conv1(x)))
      x = self.pool(torch.relu(self.conv2(x)))
      x = self.pool(torch.relu(self.conv3(x)))
      x = self.pool(torch.relu(self.conv4(x)))
      x = self.pool(torch.relu(self.conv5(x)))

      # Flatten
      x = torch.flatten(x, start_dim=1)

      # Classifier
      x = torch.relu(self.fc1(x))
      x = torch.relu(self.fc2(x))
      x = self.fc3(x)

      return x
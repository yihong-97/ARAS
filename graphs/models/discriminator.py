import torch.nn as nn
import torch.nn.functional as F


class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, num_classes//2, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(num_classes//2, num_classes//4, kernel_size=3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(num_classes//4, num_classes//8, kernel_size=3, stride=1, padding=1)
		self.classifier = nn.Conv2d(num_classes//8, 1, kernel_size=3, stride=1, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		x_s = self.sigmoid(x)

		return x,x_s

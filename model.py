import pandas as pd
import numpy as np
import torch
import torch.nn as nn # 各种层类型的实现
import torch.nn.functional as ff # 各中层函数的实现，与层类型对应，如：卷积函数、池化函数、归一化函数等等
import torch.optim as opt # 实现各种优化算法的包
import torch.nn.init as init # 初始化参数


class model_cnn(nn.Module):
	def __init__(self):
		super(model_cnn,self).__init__()
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3,stride=1,padding=1)
		self.norm1 = nn.BatchNorm2d(32,momentum=0.1)
		self.conv1_drop = nn.Dropout2d(0.25) # 14*14*32


		self.conv2 = nn.Conv2d(32, 32, kernel_size=3,stride=1,padding=1)
		self.norm2 = nn.BatchNorm2d(32,momentum=0.1)
		self.conv2_drop = nn.Dropout2d(0.25) # 14*14*32

		self.conv3 = nn.Conv2d(32, 64, kernel_size=3,stride=1,padding=1)
		self.norm3 = nn.BatchNorm2d(64,momentum=0.1)
		self.conv3_drop = nn.Dropout2d(0.25) # 14*14*32


		self.conv4 = nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1)
		self.norm4 = nn.BatchNorm2d(64,momentum=0.1)
		self.conv4_drop = nn.Dropout2d(0.25) # 14*14*32


		self.conv5 = nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=1)
		self.norm5 = nn.BatchNorm2d(128,momentum=0.1)
		self.conv5_drop = nn.Dropout2d(0.25) # 14*14*32


		self.conv6 = nn.Conv2d(128, 128, kernel_size=3,stride=1,padding=1)
		self.norm6 = nn.BatchNorm2d(128,momentum=0.1)
		self.conv6_drop = nn.Dropout2d(0.25) # 14*14*32

		self.fc1 = nn.Linear(7*7*128, 3096)
		self.norm_1 = nn.BatchNorm1d(3096,momentum=0.1)

		self.fc2 = nn.Linear(3096, 256)
		self.norm_2 = nn.BatchNorm1d(256,momentum=0.1)
		
		self.fc3 = nn.Linear(256, 10)
		
		self.fc_drop = nn.Dropout(0.5) # 14*14*32

		init.xavier_uniform_(self.conv1.weight,gain=1)
		init.constant_(self.conv1.bias, 0.1)
		init.xavier_uniform_(self.conv2.weight,gain=1)
		init.constant_(self.conv2.bias, 0.1)
		init.xavier_uniform_(self.conv3.weight,gain=1)
		init.constant_(self.conv3.bias, 0.1)
		init.xavier_uniform_(self.conv4.weight,gain=1)
		init.constant_(self.conv4.bias, 0.1)
		init.xavier_uniform_(self.conv5.weight,gain=1)
		init.constant_(self.conv5.bias, 0.1)
		init.xavier_uniform_(self.conv6.weight,gain=1)
		init.constant_(self.conv6.bias, 0.1)

	def forward(self, x):
		cuda=torch.cuda.is_available()
		device = torch.device("cuda" if cuda else "cpu")

		x = self.conv1(x)
		x = ff.relu(self.norm1(x))
		#x = self.conv1_drop(x)
		
		x = self.conv2(x)
		x = ff.relu(self.norm2(x))

		x = ff.max_pool2d(x, 2, 2)
		x = self.conv2_drop(x)
		
		x = self.conv3(x)
		x = ff.relu(self.norm3(x))
		#x = self.conv3_drop(x)

		x = self.conv4(x)
		x = ff.relu(self.norm4(x))

		x = ff.max_pool2d(x, 2, 2)
		x = self.conv4_drop(x)

		x = self.conv5(x)
		x = ff.relu(self.norm5(x))

		x = self.conv6(x)
		x = ff.relu(self.norm6(x))
		x = self.conv6_drop(x)

		x = x.view(-1, 7*7*128) # batch size and feature map size
		x = self.norm_1(ff.relu(self.fc1(x)))
		x = self.fc_drop(self.norm_2(ff.relu(self.fc2(x))))
		#x = ff.relu(self.fc2(x))
		output = self.fc3(x)
		return output

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

def swap(a,b):
	return b,a
class TEST(data.Dataset):
	def __init__(self, dataset, labelset):
		self.dataset = dataset
		self.labelset = labelset
		self.trans = transforms.Compose([transforms.ToTensor()])

	def __getitem__(self, index):
		label=self.labelset[index][0]
		data=self.dataset[index,:].reshape([28,28])
		data=self.trans(data).float()
		data=data/255.0
		return (data,label)

	def __len__(self):
		return self.dataset.shape[0]

#dataset = MyDataset(images, labels)

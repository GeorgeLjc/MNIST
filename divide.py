import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import pandas as pd

def train_valid(all_size):
	valid_size=all_size//10 #int(all_size/30)
	while(True):
		valid_indx=np.random.randint(0,all_size,valid_size)
		tmp=np.array([0 for i in range(all_size)])
		tmp[valid_indx]=1
		if(sum(tmp)%100==0):
			break

	train_index=np.array([1 for i in range(all_size)])
	train_index[valid_indx]=0
	return (train_index,tmp)





all_data=pd.read_csv('train.csv')
all_headers = np.array(all_data.columns.values)
all_data=np.array(all_data.values)
all_size=all_data.shape[0]

train_index,valid_index=train_valid(all_size)
trian_set=all_data[train_index==1,:]
valid_set=all_data[valid_index==1,:]
print(trian_set.shape[0],valid_set.shape[0])

#write prediction into a csv file
trainframe=pd.DataFrame(trian_set,columns=all_headers)

#index=False means no row index will show in csv file.
trainframe.to_csv("trainset.csv",index=False,sep=',')

#write prediction into a csv file
validframe=pd.DataFrame(valid_set,columns=all_headers)

#index=False means no row index will show in csv file.
validframe.to_csv("validset.csv",index=False,sep=',')
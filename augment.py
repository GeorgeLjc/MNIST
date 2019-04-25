import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dataset=pd.read_csv('trainset.csv')
headers=np.array(dataset.columns.values)
dataset=np.array(dataset.values)

labelset=dataset[:,0].reshape([-1,1])
dataset=np.uint8(dataset[:,1:])
img_num=dataset.shape[0]
img_size=dataset.shape[1]
exp_dataset=[]
exp_labelset=[]
for i in range(img_num):
	exp_dataset.append(dataset[i].tolist())
	exp_labelset.append(labelset[i])

	right_shift=np.copy(dataset[i])
	for j in range(img_size-4,0,-1):
		if((j%28)<=24):
			right_shift[j+3]=right_shift[j]
		if((j%28)<=2):
			right_shift[j]=0
	exp_dataset.append(right_shift.tolist())
	exp_labelset.append(labelset[i])

	left_shift=np.copy(dataset[i])
	for j in range(0,img_size):
		if((j%28)>=3):
			left_shift[j-3]=left_shift[j]
		if((j%28)>=25):
			left_shift[j]=0
	exp_dataset.append(left_shift.tolist())
	exp_labelset.append(labelset[i])

	left_rotate=np.uint8(np.copy(dataset[i]).reshape([28,28]))
	left_rotate=Image.fromarray(left_rotate,'L')
	left_rotate=np.asarray(transforms.functional.rotate(left_rotate,-10))
	left_rotate=left_rotate.reshape(-1)
	exp_dataset.append(left_rotate.tolist())
	exp_labelset.append(labelset[i])

	right_rotate=np.uint8(np.copy(dataset[i]).reshape([28,28]))
	right_rotate=Image.fromarray(right_rotate,'L')
	right_rotate=np.asarray(transforms.functional.rotate(right_rotate,10))
	right_rotate=right_rotate.reshape(-1)
	exp_dataset.append(right_rotate.tolist())
	exp_labelset.append(labelset[i])
exp_dataset=np.uint8(np.array(exp_dataset))
exp_labelset=np.array(exp_labelset).reshape(-1,1)
dataset=np.concatenate((exp_labelset,exp_dataset),axis=1)
print("dataset shape",dataset.shape)

#write prediction into a csv file
trainframe=pd.DataFrame(dataset,columns=headers)

#index=False means no row index will show in csv file.
trainframe.to_csv("trainset_augment.csv",index=False,sep=',')

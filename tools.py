import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def valid_aug(i):
	dataset=pd.read_csv('trainset_augment.csv')
	headers=np.array(dataset.columns.values)
	dataset=np.array(dataset.values)

	tmp=[]
	img_size=dataset.shape[1]-1
	left_shift=np.copy(dataset[0,:])
	print(left_shift)
	for j in range(3,img_size):
		if((j%28)>=3):
			left_shift[j-3]=left_shift[j]

		if((j%28)>=25):
			left_shift[j]=0
	right_shift=np.copy(dataset[0,:])
	for j in range(img_size-4,0,-1):
		if((j%28)<=24):
			right_shift[j+3]=right_shift[j]
		if((j%28)<=2):
			right_shift[j]=0
	tmp.append(dataset[0].tolist())
	tmp.append(right_shift.tolist())
	#dataset=np.array(tmp)

	#write prediction into a csv file
	trainframe=pd.DataFrame(tmp,columns=headers)

	#index=False means no row index will show in csv file.
	trainframe.to_csv("shift.csv",index=False,sep=',')

	labelset=dataset[:,0].reshape([-1,1])
	dataset=np.uint8(dataset[:,1:])
	plt.imshow(dataset[i].reshape((28,28)))
	print(i,labelset[i])
	plt.show()

valid_aug(13)
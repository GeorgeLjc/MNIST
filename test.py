import pandas as pd
import numpy as np
import argparse
from model import model_cnn
from dataset import MNIST
import torch.utils.data as data
import torch

#read data from csv file
test_data=pd.read_csv('test.csv')
test_data=np.array(test_data.values)
test_size=test_data.shape[0]
test_label=np.array([[0] for i in range(test_size)])
test_id=np.array(range(1,test_size+1))

#set parameters
#whether gpu is available
cuda=torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

#create model
model = model_cnn()
if(cuda):
	model=model.to(device)

#load model
model.load_state_dict(torch.load('params.pkl'))
model.eval()

#create validation dataset
testing_dataset=MNIST(test_data,test_label)

#create train loader
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
test_loader = torch.utils.data.DataLoader(testing_dataset,batch_size=1024,shuffle=False,**kwargs)

#create model loss function
loss_func=torch.nn.CrossEntropyLoss()

current=0
prediction=[]
for index,data in enumerate(test_loader):
	img,label=data
	if cuda:
		img=img.cuda()
		label=label.cuda()
	#caculate output
	out = model(img)
	_,predic=torch.max(out,1) 
	prediction.extend(predic.tolist())
	current+=len(img)
	print('{}/{}'.format(current,len(test_loader.dataset)))
prediction=np.array(prediction)

#write prediction into a csv file
dataframe=pd.DataFrame({'ImageId':test_id,'Label':prediction})

#index=False means no row index will show in csv file.
dataframe.to_csv("prediction.csv",index=False,sep=',')



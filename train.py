import pandas as pd
import numpy as np
import argparse
from model import model_cnn
from dataset import MNIST
import torch.utils.data as data
import torch

#read training data from csv file
train_data=pd.read_csv('trainset_augment.csv')
train_data=np.array(train_data.values)
train_target=train_data[:,0].reshape([-1,1])
train_data=train_data[:,1:]
print("training set size : ",train_data.shape[0])

#get the size of csv file
train_size=train_data.shape[0]


#get the validation set and validation set
validation_data=pd.read_csv('validset.csv')
validation_data=np.array(validation_data.values)
validation_target=validation_data[:,0].reshape([-1,1])
validation_data=validation_data[:,1:]
print("validation set size : ",validation_data.shape[0])

#get parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
					help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
					help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
					help='SGD momentum (default: 0.5)')
parser.add_argument('--epoch', type=int, default=1, metavar='EP',
					help='training epoch (default: 1)')
args = parser.parse_args()

#set epoch
epoch=args.epoch

#whether gpu is available
cuda=torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

#create model
model = model_cnn().to(device)
model.train()

#create training dataset
training_dataset=MNIST(train_data,train_target)

#create validation dataset
validation_dataset=MNIST(validation_data,validation_target)

#create train loader
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(training_dataset,batch_size=args.batch_size,shuffle=True,**kwargs)

#create validation loader
valid_loader = torch.utils.data.DataLoader(validation_dataset,batch_size=args.batch_size,shuffle=True,**kwargs)

#create optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,eps=1e-8, momentum=args.momentum)
reducer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=3,factor=0.5,min_lr=0.00001)

#create model loss function
loss_func=torch.nn.CrossEntropyLoss()


#start training
vloss=10000.0
for ep in range(epoch):
	img_count=0
	#training
	for index,data in enumerate(train_loader):
		img,label= data
		if cuda:
			img=img.cuda()
			label=label.cuda()
		#caculate output
		out = model(img)

		#caculate model loss
		training_loss=loss_func(out,label)

		#clear old gradient
		optimizer.zero_grad()
		
		#back propagation
		training_loss.backward()
		
		#update gradient
		optimizer.step()
		img_count=img_count+len(img)
		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				ep, img_count, len(train_loader.dataset),
				100. * index / len(train_loader), training_loss.item()))
	reducer.step(training_loss)	

	#validation
	acc=0.0
	correct_predict=0.0
	validation_loss=0.0
	for index,data in enumerate(valid_loader):
		img,label=data
		if cuda:
			img=img.cuda()
			label=label.cuda()

		out=model(img)
		validation_loss+=loss_func(out,label)
		# max(out,0) means return the (max number, row index) of each columns
		# max(out,1) means return thr (max number, columns index) of each row 
		_,predic=torch.max(out,1) 

		correct_predict+=(predic==label).sum()

	correct_predict=correct_predict.cpu().numpy()
	acc=(correct_predict)/len(valid_loader.dataset)
	validation_loss=validation_loss/len(valid_loader)
	print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		validation_loss, correct_predict, len(valid_loader.dataset),
		100. * acc))
	if(vloss-validation_loss>1e-6):
		torch.save(model.state_dict(), 'params.pkl')
		vloss=validation_loss




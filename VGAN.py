
import numpy as np
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms, datasets, models
from torch import Tensor

import math
import matplotlib.pyplot as plt 
import scipy

from scipy import io as sio
from scipy.io import savemat
from scipy.io import loadmat



class G(nn.Module):

	def __init__(self, G_in, G_out):
		super(G, self).__init__()

		self.fc1 = nn.Linear(G_in, 512)
		self.fc2 = nn.Linear(512, 512)
		self.fc3 = nn.Linear(512, 512)
		self.out = nn.Linear(512, G_out)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.out(x)

		return x


class D(nn.Module):

	def __init__(self, D_in, D_out):
		super(D, self).__init__()

		self.fc1 = nn.Linear(D_in, 512)
		self.fc2 = nn.Linear(512, 512)
		self.fc3 = nn.Linear(512, 512)
		self.out = nn.Linear(512, D_out)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.sigmoid(self.out(x))

		return x


class speechdata(Dataset):
	def __init__(self, train_path):
		self.path = train_path
		self.files = listdir(train_path)
		self.length = len(self.files)

	def __getitem__(self, index):
		var = loadmat(join(self.path, self.files[int(index)]))
		return np.array(var['Feat']), np.array(var['Clean_cent'])

	def __len__(self):
		return self.length


def training(data_loader, no_epoch):

	gnet.train()
	dnet.train()

	for en, (ip, real) in enumerate(data_loader):
		ip = Variable(ip.squeeze(0))
		real = Variable(real.squeeze(0))

		valid = Variable(Tensor(ip.size(0), 1).fill_(1.0), requires_grad=False)
		fake = Variable(Tensor(ip.size(0), 1).fill_(0.0), requires_grad=False)

		optimizerG.zero_grad()

		gout = gnet(ip)

		loss_G = adversarial_loss(dnet(gout), valid)
		loss_G.backward()
		optimizerG.step()

		optimizerD.zero_grad()	

		real_loss = adversarial_loss(dnet(real), valid)
		fake_loss = adversarial_loss(dnet(gout.detach()), fake)
	
		loss_D = (real_loss + fake_loss) / 2
		loss_D.backward()
		optimizerD.step()

		print("Epoch: {} Batch: {} Loss_Generator: {} Loss_Discriminator: {}".format(no_epoch, en, loss_G, loss_D))
		

def validating(data_loader):

	gnet.eval()
	dnet.eval()

	countG = 0
	countD = 0

	for en, (ip, real) in enumerate(data_loader):
		ip = Variable(ip.squeeze(0))
		real = Variable(real.squeeze(0))

		valid = Variable(Tensor(ip.size(0), 1).fill_(1.0), requires_grad=False)
		fake = Variable(Tensor(ip.size(0), 1).fill_(1.0), requires_grad=False)

		gout = gnet(ip)

		loss_G = adversarial_loss(dnet(gout), valid)
		countG += loss_G

		real_loss = adversarial_loss(dnet(dnet(real), valid))
		fake_loss = adversarial_loss(dnet(gout.detach()), fake)

		loss_D = (real_loss + fake_loss) / 2
		countD += loss_D

		print("Loss_Generator: {} Loss_Discriminator: {}".format(loss_G, loss_D))

	return countG, countD





training_path = speechdata(train_path="/home/speechlab/First/Training_complementary_feats")
train_data = DataLoader(dataset=training_path, batch_size=1, shuffle=True, num_workers=2)

validating_path = speechdata(valid_path="/home/speechlab/First/")
valid_data = DataLoader(dataset=validating_path, batch_size=1, shuffle=True, num_workers=2)


gnet = G(40, 40)
dnet = D(40, 1)

adversarial_loss = nn.BCELoss() 

optimizerG = optim.Adam(gnet.parameters(), lr=0.001)
optimizerD = optim.Adam(dnet.parameters(), lr=0.001)


epoch = 5

valid_array_G = []
valid_array_D = []

for iterate in range(epoch):
	
	training(train_data, iterate+1)

	if (iterate+1)%5==0:
		torch.save(gnet, join(mainfolder,"gen_Ep_{}.pth".format(iterate+1)))
		torch.save(dnet, join(mainfolder,"dis_Ep_{}.pth".format(iterate+1)))


	valid_c_G, valid_c_D = validating(valid_data)
	valid_c_G.append(valid_c_G)
	valid_c_D.append(valid_c_D)


mainfolder="/home/speechlab/First/Trial_5_Saved"

savemat(mainfolder+"/"+str('discriminator_loss.mat'),  mdict={'foo': arG})
savemat(mainfolder+"/"+str('generator_loss.mat'),  mdict={'foo': arD})













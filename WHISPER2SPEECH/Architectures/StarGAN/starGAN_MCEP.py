
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
import torch

import visdom
import math
import matplotlib.pyplot as plt 
import scipy
import h5py
from scipy import io as sio
from scipy.io import savemat
from scipy.io import loadmat

# from thop import profile

viz = visdom.Visdom()

print("\n\n\n\n\nCuda available:",torch.cuda.is_available(),"\n\n\n\n\n")


# -------------------- Data -----------------------------------------------------------------------------------------------------------------------------

class speech_data(Dataset):
    
    def __init__(self, folder_path, mainfolder, train=True):
        self.path = folder_path
        self.folds = listdir(self.path)
        self.mainfolder = mainfolder
        self.train = train
        
    def __getitem__(self, index):
        while True:
            self.c_org,self.c_trg = torch.LongTensor(2).random_(0, 4)
            if(self.c_org != self.c_trg):
                break
        
        fd1 = join(join(self.path, self.folds[self.c_org]),self.mainfolder)
        fd2 = join(join(self.path, self.folds[self.c_trg]), self.mainfolder)

        dd1 = listdir(fd1)
        dd2 = listdir(fd2)
        
        id1 = random.randint(0,len(fd1))
        id2 = random.randint(0,len(fd2))
        
        d1 = loadmat(join(fd1,dd1[id1]))
        d2 = loadmat(join(fd2,dd2[id2]))

        return np.array(d1['Clean_cent']), np.array(d1['Feat']), np.array(d2['Clean_cent']), np.array(d2['Feat']), self.c_org, self.c_trg_index        #[:,125:150]
    
    def __len__(self):
        if self.train:
            return 5000
        else:
            return 100

class test_speech_data(Dataset):
    
    def __init__(self, folder_path):
        self.path = folder_path
        self.files = listdir(folder_path)
        self.length = len(self.files)
        
    def __getitem__(self, index):
        # result = np.zeros((1000,275))
        d = loadmat(join(self.path, self.files[int(index)]))
        print(index)
        # result[:d.shape[0],:d.shape[1]] = d 
        return np.array(d['Feat'])#[:,125:150]
    
    def __len__(self):
        return self.length
        
mainfolder = "/media/maitreya/Maitreya_Hard_Disk/Speech_Lab_VC/INTERSPEECH19-paper2/star-gan/result/US_103/MCEP/model"

traindata = speech_data(folder_path="/media/maitreya/Maitreya_Hard_Disk/Speech_Lab_VC/INTERSPEECH19-paper2/data", mainfolder="WHSP2SPCH_MCEP/batches/Training_complementary_feats")
train_dataloader = DataLoader(dataset=traindata, batch_size=1, shuffle=True, num_workers=2)


# -------------------- Network -----------------------------------------------------------------------------------------------------------------------------

class generator(nn.Module):
    
    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.out.weight)

        nn.init.xavier_uniform_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.bias)
        nn.init.xavier_uniform_(self.out.bias)

    def __init__(self, G_in, G_out,no_of_domains, w1, w2, w3):
        super(generator, self).__init__()
        
        self.fc1= nn.Linear(G_in+no_of_domains, w1)
        self.fc2= nn.Linear(w1, w2)
        self.fc3= nn.Linear(w2, w3)
        self.out= nn.Linear(w3, G_out)

        # self.weight_init()
        
    def forward(self, x,c):
        
        #c.view(1,c.size(0))	#c: 1x10
        c = c.repeat(x.size(0),1)	#c:1000x10
        x = torch.cat([x, c], dim=1)	#concatening x with c
		
        # x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        # x = x.view(1, 1, 1000, 25)
        return x
        

class discriminator(nn.Module):

    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.out.weight)

        nn.init.xavier_uniform_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.bias)
        nn.init.xavier_uniform_(self.out.bias)

    def __init__(self, D_in, D_out, no_of_domains, w1, w2, w3):
        super(discriminator, self).__init__()
        
        self.fc1= nn.Linear(D_in, w1)
        self.fc2= nn.Linear(w1, w2)
        self.fc3= nn.Linear(w2, w3)
        self.out= nn.Linear(w3, D_out)
        self.cls= nn.Linear(w3, no_of_domains)

        # self.weight_init()
        
    def forward(self, y):
        
        # y = y.view(y.size(0), -1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y_src = F.sigmoid(self.out(y))	#y_src is for real/fake; dim=1
        y_class = self.cls(y)	#y_class is for classifier probability ;dim= no_of_domains
        return y_src,y_class


# -------------------- Inintializing Parameters -----------------------------------------------------------------------------------------------------------------------------

adversarial_loss = nn.BCELoss()
mmse_loss = nn.MSELoss()
classifier_loss = nn.CrossEntropyLoss()

no_of_domains = 4	#no of multi-domains needed in starGAN

Gnet = generator(40, 40,no_of_domains, 512, 512, 512).cuda()
Dnet = discriminator(40, 1,no_of_domains , 512, 512, 512).cuda()

# flops1, params1 = profile(Gnet_sw, input_size=(1000,40))
# print(flops1, params1)
# flops2, params2 = profile(Gnet_ws, input_size=(1000,40))
# print(flops2, params2)

# Optimizers
optimizer_G = torch.optim.Adam(Gnet.parameters(), lr=0.0001)
optimizer_D = torch.optim.Adam(Dnet.parameters(), lr=0.0001)


# -------------------- Training -----------------------------------------------------------------------------------------------------------------------------

def training(data_loader, n_epochs):
    Gnet.train()
    Dnet.train()
    
    for en, (b_org, a_org,b_trg,a_trg,c_org_index,c_trg_index) in enumerate(data_loader):
        a_org = Variable(a_org.squeeze(0)).cuda()
        b_org = Variable(b_org.squeeze(0)).cuda()
        a_trg = Variable(a_trg.squeeze(0)).cuda()
        b_trg = Variable(b_trg.squeeze(0)).cuda()
        
        c_org = torch.zeros(1,no_of_domains)
        c_org[0,c_org_index] = 1
        c_trg = torch.zeros(1,no_of_domains)
        c_trg[0,c_trg_index] = 1

        c_org = Variable(c_org).cuda()
        c_trg = Variable(c_trg).cuda()

        c_trg_index = Variable(c_trg_index).cuda()
        c_org_index = Variable(c_org_index).cuda()

        valid = Variable(Tensor(a_org.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
        fake = Variable(Tensor(a_org.shape[0], 1).fill_(0.0), requires_grad=False).cuda()

        optimizer_G.zero_grad()
        Gout_c_trg = Gnet(a_org,c_trg)
        G_loss_adv_t = adversarial_loss(Dnet(Gout_c_trg)[0], valid)			#adversial loss
        G_loss_adv_s = adversarial_loss(Dnet(b_org)[0], fake)

        G_loss_cyc = mmse_loss(Gnet(Gout_c_trg,c_org),b_org)*1	#cycle loss		
		
        G_loss_id = mmse_loss(Gnet(a_trg,c_trg),b_trg)*1		#identity loss
        
        G_loss = G_loss_adv_s + G_loss_adv_t + G_loss_cyc + G_loss_id
        G_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        # print(c_trg_index.repeat(1000,1).shape,Dnet(b_trg)[1].shape)
        real_loss = adversarial_loss(Dnet(b_trg)[0], valid)*1 + classifier_loss(Dnet(b_trg)[1],c_trg_index.repeat(1000))*1 
        fake_loss = adversarial_loss(Dnet(Gout_c_trg.detach())[0], fake)*1 + classifier_loss(Dnet(Gout_c_trg.detach())[1],c_trg_index.repeat(1000))*1  
        D_loss = (real_loss + fake_loss) / 2

        D_loss.backward()
        optimizer_D.step()
        
        print ("[Epoch: %d] [Iter: %d/%d] [D loss: %f] [G loss: %f]" % (n_epochs, en, len(data_loader), D_loss.cpu().data.numpy(), G_loss.cpu().data.numpy()))


# def validation(data_loader):
   
#     Gnet.eval()
#     Dnet.eval()
    
#     for en, (b_org, a_org,b_trg,a_trg,c_org_index,c_trg_index) in enumerate(data_loader):
#         a_org = Variable(a_org.squeeze(0)).cuda()
#         b_org = Variable(b_org.squeeze(0)).cuda()
#         a_trg = Variable(a_trg.squeeze(0)).cuda()
#         b_trg = Variable(b_trg.squeeze(0)).cuda()
        
#         c_org = torch.zeros(1,no_of_domains)
#         c_org[0,c_org_index] = 1
#         c_trg = torch.zeros(1,no_of_domains)
#         c_trg[0,c_trg_index] = 1

#         c_org = Variable(c_org).cuda()
#         c_trg = Variable(c_trg).cuda()

#         c_trg_index = Variable(c_trg_index).cuda()
#         c_org_index = Variable(c_org_index).cuda()

#         valid = Variable(Tensor(a_org.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
#         fake = Variable(Tensor(a_org.shape[0], 1).fill_(0.0), requires_grad=False).cuda()
		
        
#         Gout_c_trg = Gnet(a_org,c_trg)
#         G_loss_adv_t = adversarial_loss(Dnet(Gout_c_trg)[0], valid)			#adversial loss
#         G_loss_adv_s = adversarial_loss(Dnet(b_org)[0], fake)

#         G_loss_cyc = mmse_loss(Gnet(Gout_c_trg,c_org),b_org)*1	#cycle loss		
		
#         G_loss_id = mmse_loss(Gnet(a_trg,c_trg),b_trg)*1		#identity loss
        
#         G_loss = G_loss_adv_s + G_loss_adv_t + G_loss_cyc + G_loss_id
        
#         real_loss = adversarial_loss(Dnet(b_trg)[0], valid)*1 + classifier_loss(Dnet(b_trg)[1],c_trg_index.repeat(1000))*1 
#         fake_loss = adversarial_loss(Dnet(Gout_c_trg.detach())[0], fake)*1 + classifier_loss(Dnet(Gout_c_trg.detach())[1],c_trg_index.repeat(1000))*1  
#         D_loss = (real_loss + fake_loss) / 2

#         # print(en)
#     return G_loss, D_loss

 
# -----------------------------------------------------------------------------------------------------------------------------------------------------------    
 
isTrain = False

if isTrain:
    epoch = 100
    dl_arr = []
    gl_arr = []
    for ep in range(epoch):

        training(train_dataloader, ep+1)
        if (ep+1)%5==0:
            torch.save(Gnet, join(mainfolder,"gen_g_{}_d_{}_Ep_{}.pth".format(5,1,ep+1)))
            torch.save(Dnet, join(mainfolder,"dis_g_{}_d_{}_Ep_{}.pth".format(5,1,ep+1)))
        # dl,gl = validation(val_dataloader)
        # print("D_loss: " + str(dl) + " G_loss: " + str(gl))
        # dl_arr.append(dl)
        # gl_arr.append(gl)
        # if ep == 0:
        #     gplot = viz.line(Y=np.array([gl]), X=np.array([ep]), opts=dict(title='Generator'))
        #     dplot = viz.line(Y=np.array([dl]), X=np.array([ep]), opts=dict(title='Discriminator'))
        # else:
        #     viz.line(Y=np.array([gl]), X=np.array([ep]), win=gplot, update='append')
        #     viz.line(Y=np.array([dl]), X=np.array([ep]), win=dplot, update='append')

            
    # savemat(mainfolder+"/"+str('discriminator_loss.mat'),  mdict={'foo': dl_arr})
    # savemat(mainfolder+"/"+str('generator_loss.mat'),  mdict={'foo': dl_arr})

    # plt.figure(1)
    # plt.plot(dl_arr)
    # plt.savefig(mainfolder+'/discriminator_loss.png')
    # plt.figure(2)
    # plt.plot(gl_arr)
    # plt.savefig(mainfolder+'/generator_loss.png')


# -------------------- Testing -----------------------------------------------------------------------------------------------------------------------------

else:
    print("Testing")
    save_folder = "/media/maitreya/Maitreya_Hard_Disk/Speech_Lab_VC/INTERSPEECH19-paper2/star-gan/result/US_106/MCEP/mask/"
    test_folder_path="/media/maitreya/Maitreya_Hard_Disk/Speech_Lab_VC/INTERSPEECH19-paper2/data/US_106/WHSP2SPCH_MCEP/batches/Testing_complementary_feats/"
    n = len(listdir(test_folder_path))
    Gnet = torch.load(join(mainfolder,"gen_g_5_d_1_Ep_70.pth"))

    for i in range(n):
        d = loadmat(join(test_folder_path, "Test_Batch_{}.mat".format(str(i))))
        a = torch.from_numpy(d['Feat'])
        a = Variable(a.squeeze(0).type('torch.FloatTensor')).cuda()
        c_trg = torch.zeros(1,no_of_domains).cuda()
        c_trg[0,3]=1
        Gout = Gnet(a,c_trg)
        # np.save(join(save_folder,'Test_Batch_{}.npy'.format(str(i))), Gout.cpu().data.numpy())
        savemat(join(save_folder,'Test_Batch_{}.mat'.format(str(i))),  mdict={'foo': Gout.cpu().data.numpy()})
            

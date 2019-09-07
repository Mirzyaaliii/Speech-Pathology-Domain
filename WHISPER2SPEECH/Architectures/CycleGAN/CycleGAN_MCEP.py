
import numpy as np
import sys
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
import itertools

import torchvision
from torchvision import transforms, datasets, models
from torch import Tensor

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

class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

class speech_data(Dataset):
    
    def __init__(self, folder_path):
        self.path = folder_path
        self.files = listdir(folder_path)
        self.length = len(self.files)
        
    def __getitem__(self, index):

        idx2 = index
        while(idx2==index):
            idx2 = random.randint(0, self.length)

        d1 = loadmat(join(self.path, self.files[int(index)]))
        d2 = loadmat(join(self.path, self.files[int(index)]))

        return np.array(d1['Feat']), np.array(d2['Clean_cent'])
    
    def __len__(self):
        return self.length

Path where you want to store your results        
mainfolder = "/media/maitreya/Maitreya_Hard_Disk/Speech_Lab_VC/INTERSPEECH19-paper2/old-cycleGAN/result/"+sys.argv[1] +"/MCEP/model/"

# Training Data path
traindata = speech_data(folder_path="/media/maitreya/Maitreya_Hard_Disk/Speech_Lab_VC/INTERSPEECH19-paper2/data/"+sys.argv[1]+"/WHSP2SPCH_MCEP/batches/Training_complementary_feats")
train_dataloader = DataLoader(dataset=traindata, batch_size=1, shuffle=True, num_workers=2)

# Path for validation data
valdata = speech_data(folder_path="/media/maitreya/Maitreya_Hard_Disk/Speech_Lab_VC/INTERSPEECH19-paper2/data/"+sys.argv[1]+"/WHSP2SPCH_MCEP/batches/Validation_complementary_feats")
val_dataloader = DataLoader(dataset=valdata, batch_size=1, shuffle=True, num_workers=2)


# -------------------- Network -----------------------------------------------------------------------------------------------------------------------------


class generator(nn.Module):
    
    # Weight Initialization [we initialize weights here]
    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.out.weight)
        
        nn.init.xavier_uniform_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.bias)
        nn.init.xavier_uniform_(self.out.bias)
    
    def __init__(self, G_in, G_out, w1, w2, w3):
        super(generator, self).__init__()
        
        self.fc1= nn.Linear(G_in, w1)
        self.fc2= nn.Linear(w1, w2)
        self.fc3= nn.Linear(w2, w3)
        self.out= nn.Linear(w3, G_out)
    
    # Deep neural network [you are passing data layer-to-layer]
    def forward(self, x):
        
        # x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        
        return x


class discriminator(nn.Module):
    
    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.out.weight)
    
    def __init__(self, D_in, D_out, w1, w2, w3):
        super(discriminator, self).__init__()
        
        self.fc1= nn.Linear(D_in, w1)
        self.fc2= nn.Linear(w1, w2)
        self.fc3= nn.Linear(w2, w3)
        self.out= nn.Linear(w3, D_out)
    
    # self.weight_init()
    
    def forward(self, y):
        
        # y = y.view(y.size(0), -1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = F.sigmoid(self.out(y))
        return y


# -------------------- Inintializing Parameters -----------------------------------------------------------------------------------------------------------------------------

# Loss Functions
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

#I/O variables
in_g = 40
out_g = 40
in_d = 40
out_d = 1
learning_rate = 0.0001


Gnet_ws = generator(in_g, out_g, 512, 512, 512).cuda()
Gnet_sw = generator(in_g, out_g, 512, 512, 512).cuda()
Dnet_w = discriminator(in_d, out_d, 512, 512, 512).cuda()
Dnet_s = discriminator(in_d, out_d, 512, 512, 512).cuda()

# flops1, params1 = profile(Gnet_sw, input_size=(1000,40))
# print(flops1, params1)
# flops2, params2 = profile(Gnet_ws, input_size=(1000,40))
# print(flops2, params2)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(Gnet_ws.parameters(), Gnet_sw.parameters()),lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_w = torch.optim.Adam(Dnet_w.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_s = torch.optim.Adam(Dnet_s.parameters(), lr=learning_rate, betas=(0.5, 0.999))


# -------------------- Training -----------------------------------------------------------------------------------------------------------------------------

# Training Function
def training(data_loader, n_epochs):
    Gnet_ws.train()
    Gnet_sw.train()
    Dnet_w.train()
    Dnet_s.train()
    
    for en, (a, b) in enumerate(data_loader):
        a = Variable(a.squeeze(0)).cuda()
        b = Variable(b.squeeze(0)).cuda()

        valid = Variable(Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
        fake = Variable(Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False).cuda()
        
        ###### Generators W2S and S2W ######
        optimizer_G.zero_grad()
        
        # Identity loss
        # G_W2S(S) should equal S if real S is fed
        same_s = Gnet_ws(b)
        loss_identity_s = criterion_identity(same_s, b)*5.0
        # G_S2W(W) should equal W if real W is fed
        same_w = Gnet_sw(a)
        loss_identity_w = criterion_identity(same_w, a)*5.0

        # GAN loss
        Gout_ws = Gnet_ws(a)
        loss_GAN_W2S = criterion_GAN(Dnet_s(Gout_ws), valid)
        
        Gout_sw = Gnet_sw(b)
        loss_GAN_S2W = criterion_GAN(Dnet_w(Gout_sw), valid)
        
        # Cycle loss
        recovered_W = Gnet_sw(Gout_ws)
        loss_cycle_WSW = criterion_cycle(recovered_W, a)*10.0
        
        recovered_S = Gnet_ws(Gout_sw)
        loss_cycle_SWS = criterion_cycle(recovered_S, b)*10.0
        
        # Total loss
        loss_G =  loss_identity_w + loss_identity_s + loss_GAN_W2S + loss_GAN_S2W + loss_cycle_WSW + loss_cycle_SWS
        loss_G.backward()
        
        optimizer_G.step()
        
        
        ###### Discriminator W ######
        optimizer_D_w.zero_grad()

        # Real loss
        loss_D_real = criterion_GAN(Dnet_w(a), valid)
        
        # Fake loss
        loss_D_fake = criterion_GAN(Dnet_w(Gout_sw.detach()), fake)
        
        # Total loss
        loss_D_w = (loss_D_real + loss_D_fake)*0.5
        loss_D_w.backward()
        
        optimizer_D_w.step()
        
        ###################################
        
        ###### Discriminator B ######
        optimizer_D_s.zero_grad()
        
        # Real loss
        loss_D_real = criterion_GAN(Dnet_s(b), valid)
        
        # Fake loss
        loss_D_fake = criterion_GAN(Dnet_s(Gout_ws.detach()), fake)
        
        # Total loss
        loss_D_s = (loss_D_real + loss_D_fake)*0.5
        loss_D_s.backward()
        
        optimizer_D_s.step()

        ###################################
        
        # D_loss = 0

        #D_running_loss = 0
        #D_running_loss += D_loss.item()
        
        print ("[Epoch: %d] [Iter: %d/%d] [D_S loss: %f] [D_W loss: %f] [G loss: %f]" % (n_epochs, en, len(data_loader), loss_D_s, loss_D_w, loss_G.cpu().data.numpy()))
    


def validating(data_loader):
    Gnet_ws.eval()
    Gnet_sw.eval()
    Dnet_s.eval()
    Dnet_w.eval()
    Grunning_loss = 0
    Drunning_loss = 0

    
    for en, (a, b) in enumerate(data_loader):
        a = Variable(a.squeeze(0)).cuda()
        b = Variable(b.squeeze(0)).cuda()
        
        valid = Variable(Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
        fake = Variable(Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False).cuda()

        ###### Generators W2S and S2W ######
        
        # Identity loss
        # G_W2S(S) should equal S if real S is fed
        same_s = Gnet_ws(b)
        loss_identity_s = criterion_identity(same_s, b)*5.0
        # G_S2W(W) should equal W if real W is fed
        same_w = Gnet_sw(a)
        loss_identity_w = criterion_identity(same_w, a)*5.0
        
        # GAN loss
        Gout_ws = Gnet_ws(a)
        loss_GAN_W2S = criterion_GAN(Dnet_s(Gout_ws), valid)
        
        Gout_sw = Gnet_sw(b)
        loss_GAN_S2W = criterion_GAN(Dnet_w(Gout_sw), valid)
        
        # Cycle loss
        recovered_W = Gnet_sw(Gout_ws)
        loss_cycle_WSW = criterion_cycle(recovered_W, a)*10.0
        
        recovered_S = Gnet_ws(Gout_sw)
        loss_cycle_SWS = criterion_cycle(recovered_S, b)*10.0
        
        # Total loss
        loss_G =  loss_identity_w + loss_identity_s + loss_GAN_W2S + loss_GAN_S2W + loss_cycle_WSW + loss_cycle_SWS

        
        ###### Discriminator W ######
        
        # Real loss
        loss_D_real = criterion_GAN(Dnet_w(a), valid)
        
        # Fake loss
        loss_D_fake = criterion_GAN(Dnet_w(Gout_sw.detach()), fake)
        
        # Total loss
        loss_D_w = (loss_D_real + loss_D_fake)*0.5

        
        ###### Discriminator B ######
        
        # Real loss
        loss_D_real = criterion_GAN(Dnet_s(b), valid)
        
        # Fake loss
        loss_D_fake = criterion_GAN(Dnet_s(Gout_ws.detach()), fake)
        
        # Total loss
        loss_D_s = (loss_D_real + loss_D_fake)*0.5
 
        ###################################
        loss_D = loss_D_s + loss_D_w	

        Grunning_loss += loss_G.item()

        Drunning_loss += loss_D.item()
        
    return Drunning_loss/(en+1),Grunning_loss/(en+1)

    
# -----------------------------------------------------------------------------------------------------------------------------------------------------------    
 
if(sys.argv[2] == "True"):
    isTrain = True
else:
    isTrain = False


if isTrain:
    epoch = 100
    dl_arr = []
    gl_arr = []
    for ep in range(epoch):

        training(train_dataloader, ep+1)
        if (ep+1)%5==0:
            torch.save(Gnet_ws, join(mainfolder,"gen_ws_Ep_{}.pth".format(ep+1)))
        #torch.save(Dnet, join(mainfolder,"dis_g_{}_d_{}_Ep_{}.pth".format(1,1,ep+1)))
        dl,gl = validating(val_dataloader)
        print("D_loss: " + str(dl) + " G_loss: " + str(gl))
        dl_arr.append(dl)
        gl_arr.append(gl)
        if ep == 0:
            gplot = viz.line(Y=np.array([gl]), X=np.array([ep]), opts=dict(title='Cycle_Generator'))
            dplot = viz.line(Y=np.array([dl]), X=np.array([ep]), opts=dict(title='Cycle_Discriminator'))
        else:
            viz.line(Y=np.array([gl]), X=np.array([ep]), win=gplot, update='append')
            viz.line(Y=np.array([dl]), X=np.array([ep]), win=dplot, update='append')

            
    savemat(mainfolder+"/"+str('discriminator_loss.mat'),  mdict={'foo': dl_arr})
    savemat(mainfolder+"/"+str('generator_loss.mat'),  mdict={'foo': gl_arr})

    plt.figure(1)
    plt.plot(dl_arr)
    plt.savefig(mainfolder+'/discriminator_loss.png')
    plt.figure(2)
    plt.plot(gl_arr)
    plt.savefig(mainfolder+'/generator_loss.png')


# -------------------- Testing -----------------------------------------------------------------------------------------------------------------------------

else:
    print("Testing")
    save_folder = "/media/maitreya/Maitreya_Hard_Disk/Speech_Lab_VC/INTERSPEECH19-paper2/old-cycleGAN/result/"+sys.argv[1]+"/MCEP/mask/"
    test_folder_path="/media/maitreya/Maitreya_Hard_Disk/Speech_Lab_VC/INTERSPEECH19-paper2/data/"+sys.argv[1]+"/WHSP2SPCH_MCEP/batches/Testing_complementary_feats"
    n = len(listdir(test_folder_path))
    Gnet = torch.load(join(mainfolder,"gen_ws_Ep_100.pth"))
    for i in range(n):
        d = loadmat(join(test_folder_path, "Test_Batch_{}.mat".format(str(i))))
        a = torch.from_numpy(d['Feat'])
        a = Variable(a.squeeze(0).type('torch.FloatTensor')).cuda()
        Gout = Gnet(a)
        # np.save(join(save_folder,'Test_Batch_{}.npy'.format(str(i))), Gout.cpu().data.numpy())
        savemat(join(save_folder,'Test_Batch_{}.mat'.format(str(i))),  mdict={'foo': Gout.cpu().data.numpy()})
            

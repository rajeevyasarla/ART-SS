import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import pdb
import numpy as np
import random
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
from numpy.linalg import inv
from sklearn.metrics.pairwise import cosine_similarity
import glob
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image

def kronecker(A, B):
    AB = torch.einsum("ab,cd->acbd", A, B)
    AB = AB.view(A.size(0)*B.size(0), A.size(1)*B.size(1))
    return AB
# numpy version of Linear kernel function 
def kernel_linear(X_u,X_l):
    xu_shape = X_u.shape
    xl_shape = X_l.shape
    # pdb.set_trace()
    if len(xu_shape)==2 :
        X_u_norm = X_u / X_u.norm(dim=-1).view(xu_shape[0],1)
        X_l_norm = X_l / X_l.norm(dim=-1).view(xl_shape[0],1)
        ker_t = torch.mm(X_u_norm, X_l_norm.transpose(0,1))
    elif len(xu_shape)==3 :
        X_u_norm = X_u / X_u.norm(dim=-1).view(xu_shape[0],xu_shape[1],1)
        X_l_norm = X_l / X_l.norm(dim=-1).view(xl_shape[0],xl_shape[1],1)
        ker_t = torch.bmm(X_u_norm, X_l_norm.transpose(1,2))
    elif len(xu_shape)==4 :
        X_u = X_u.reshape(xu_shape[0]*xu_shape[1],xu_shape[2],xu_shape[3])
        X_l = X_l.reshape(xl_shape[0]*xl_shape[1],xl_shape[2],xl_shape[3])
        X_u_norm = X_u / X_u.norm(dim=-1).view(xu_shape[0]*xu_shape[1],xu_shape[2],1)
        X_l_norm = X_l / X_l.norm(dim=-1).view(xl_shape[0]*xl_shape[1],xl_shape[2],1)
        X_l_norm = X_l_norm.transpose(1,2)        
        ker_t = torch.bmm(X_u_norm, X_l_norm)
        ker_t = ker_t.view(xu_shape[0],xu_shape[1],xu_shape[2],xl_shape[2])
    return ker_t
pdist_ker = nn.PairwiseDistance(p=2)
def kernel_distance(X_u,X_l):
    xu_shape = X_u.shape
    xl_shape = X_l.shape
    # pdb.set_trace()
    if len(xu_shape)==2 :
        x_l_t = X_l.repeat(xu_shape[0],1)
        x_u_t = X_u.repeat(xl_shape[0],1)
        ker_t = pdist_ker(x_u_t,x_l_t)
        ker_t = ker_t.view(xu_shape[0],xl_shape[0])
    elif len(xu_shape)==3 :
        x_l_t = X_l.repeat(1,xu_shape[1],1)
        x_u_t = X_u.repeat(1,xl_shape[1],1)
        ker_t = pdist_ker(x_u_t.transpose(1,2),x_l_t.transpose(1,2))
        ker_t = ker_t.view(xu_shape[0],xu_shape[1],xl_shape[1])
    elif len(xu_shape)==4 :
        X_u = X_u.reshape(xu_shape[0]*xu_shape[1],xu_shape[2],xu_shape[3])
        X_l = X_l.reshape(xl_shape[0]*xl_shape[1],xl_shape[2],xl_shape[3])
        x_l_t = X_l.repeat(1,xu_shape[2],1)
        x_u_t = X_u.repeat(1,xl_shape[2],1)
        ker_t = pdist_ker(x_u_t.transpose(1,2),x_l_t.transpose(1,2))
        ker_t = ker_t.view(xu_shape[0],xu_shape[1],xu_shape[2],xl_shape[2])
    return ker_t

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    distt  =torch.clamp(dist, 0.0, np.inf)
    dist[dist != dist] = 0
    return dist

# torch version of Squared Exponential kernel function 
def kernel_se(x,y,var):
    sigma_1 = 1.0
    pw = 0.6
    l_1 = torch.max(var)#.max(axis=-1).max(axis=-1).max(axis=-1)#1.0#(np.sum(mu**2))**(pw)
    d = kernel_distance(x,y)
    Ker = sigma_1**2 *torch.exp(-0.5*d/l_1**2)
    return Ker

# numpy version of Squared Exponential kernel function 
def kernel_se_np(x,y,var):
    sigma_1 = 1.0
    pw = 0.6
    l_1 = var.max(axis=-1).max(axis=-1).max(axis=-1)#1.0#(np.sum(mu**2))**(pw)
    d = cdist(x,y)**2
    Ker = sigma_1**2 * np.exp(-0.5*d/l_1**2)
    return Ker

# torch version of Rational Quadratic kernel function
def kernel_rq(x,y,var,alpha=0.5):
    sigma_1 = 1.0
    pw = 0.6
    l_1 = torch.max(var)#var.max(axis=-1).max(axis=-1).max(axis=-1)#1.0#(np.sum(mu**2))**(pw)
    d = kernel_distance(x,y)
    Ker = sigma_1**2 *(1+(0.5*d/(alpha*l_1**2)))**(-1*alpha)
    return Ker

# numpy version of Rational Quadratic kernel function 
def kernel_rq_np(x,y,var,alpha=0.5):
    sigma_1 = 1.0
    pw = 0.6
    l_1 = var.max(axis=-1).max(axis=-1).max(axis=-1)#1.0#(np.sum(mu**2))**(pw)
    d = cdist(x,y)**2
    Ker = sigma_1**2 * (1+(0.5*d/(alpha*l_1**2)))**(-1*alpha)
    return Ker



class ART(object):
    def __init__(self,num_lbl,num_unlbl,train_batch_size,version,kernel_type):
        self.num_lbl = num_lbl # number of labeled images
        self.num_unlbl = num_unlbl # number of unlabeled images
        self.z_height=32 # height of the feature map z i.e dim 2 
        self.z_width = 32 # width of the feature map z i.e dim 3
        self.z_numchnls = 32 # number of feature maps in z i.e dim 1
        self.num_nearest = 16 #number of nearest neighbors for unlabeled vector
        self.Fz_lbl = torch.zeros((self.num_lbl,self.z_numchnls,self.z_height,self.z_width),dtype=torch.float32).cuda() #Feature matrix Fzl for latent space labeled vector matrix
        self.Fz_unlbl = torch.zeros((self.num_unlbl,self.z_numchnls,self.z_height,self.z_width),dtype=torch.float32).cuda() #Feature matrix Fzl for latent space unlabeled vector matrix
        self.ker_lbl = torch.zeros((self.num_lbl,self.num_lbl)).cuda() # kernel matrix of labeled vectors
        self.ker_lbl_ang = torch.zeros((self.num_lbl,self.num_lbl)).cuda() # kernel matrix of labeled vectors
        self.ker_unlbl = torch.zeros((self.num_unlbl,self.num_lbl)).cuda() # kernel matrix of unlabeled vectors
        self.ker_unlbl_ang = torch.zeros((self.num_unlbl,self.num_lbl)).cuda() # kernel matrix of unlabeled vectors
        # self.metric_l = nn.Parameter(torch.rand((32,32),dtype=torch.float32), requires_grad=True)
        # self.metric_l = self.metric_l.cuda()
        self.sigma_noise = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        # self.metric_m = torch.matmul(self.metric_l.t(),self.metric_l)
        self.dict_lbl ={} # dictionary helpful in saving the feature vectors
        self.dict_unlbl ={} # dictionary helpful in saving the feature vectors
        #self.lambda_var = 0.0033 # factor multiplied with minimizing variance
        self.train_batch_size = train_batch_size
        self.version = version # version1 is GP SIMO model and version2 is GP MIMO model
        self.kernel_type = kernel_type
        self.KL_div = torch.nn.KLDivLoss()
        self.unlbl_sig = torch.zeros((self.num_unlbl,1)).cuda()
        self.lbl_sig = torch.zeros((self.num_lbl,1)).cuda()
        self.reject_unlbl = torch.zeros((self.num_unlbl,1)).cuda() #"1" indicates unlabeled image is not rejected and "0" indicates unlabeled image is rejected.


        self.thrsh_dist = 1000
        self.thrsh_ang = 100000
        self.k_NN = 5


        # declaring kernel function
        if kernel_type =='Linear':
            self.kernel_comp = kernel_linear
        elif kernel_type =='Squared_exponential':
            self.kernel_comp = kernel_se
        elif kernel_type =='Rational_quadratic':
            self.kernel_comp = kernel_rq

        if kernel_type =='Linear':
            self.kernel_comp_np = cosine_similarity
        elif kernel_type =='Squared_exponential':
            self.kernel_comp_np = kernel_se_np
        elif kernel_type =='Rational_quadratic':
            self.kernel_comp_np = kernel_rq_np

        # paths for loading masks which are used for aleotoric uncertainity
        self.mask_path = './masks/rain' 
        self.mask_names = glob.glob(self.mask_path + '/*.png')

        self.num_M = 4


    

    def gen_featmaps_unlbl(self,dataloader,net,device):
        print("Unlabelled: started storing feature vectors and kernel matrix")
        Mask_uncer = self.generate_mask()
        count =0
        self.reject_unlabl = torch.zeros((self.num_unlbl,1)).cuda()
        for batch_id, train_data in enumerate(dataloader):

            input_im, gt, imgid = train_data
            input_im = input_im.to(device)
            gt = gt.to(device)

            
            net.eval()
            B,N,_,_ = input_im.shape
            input_im = input_im.repeat(self.num_M,1,1,1)
            input_im = input_im + 0.1*Mask_uncer
            pred_image,zy_in = net(input_im)
            tensor_mat = zy_in.view(B,self.num_M,self.z_numchnls,self.z_height,self.z_width).mean(dim=1).data#torch.squeeze(zy_in.data)
            tensor_mat = tensor_mat.view(B,self.z_numchnls,self.z_height,self.z_width)
            sigma_uncer = zy_in.view(B,self.num_M,self.z_numchnls,self.z_height,self.z_width).std(dim=1).mean(dim=[-3,-2,-1])
            # saving latent space feature vectors 
            for i in range(B):
                if imgid[i] not in self.dict_unlbl.keys():
                    self.dict_unlbl[imgid[i]] = count
                    count += 1
                tmp_i = self.dict_unlbl[imgid[i]]
                self.Fz_unlbl[tmp_i,:,:,:] = tensor_mat[i,:,:,:].data
                self.unlbl_sig[tmp_i] = sigma_uncer[i].data
                # tensor = torch.squeeze(tensor_mat[i,:,:,:])
        X = self.Fz_unlbl.view(-1,self.z_numchnls*self.z_height*self.z_width)
        Y = self.Fz_lbl.view(-1,self.z_numchnls*self.z_height*self.z_width)
        dist = torch.from_numpy(euclidean_distances(X.cpu().numpy(),Y.cpu().numpy())).cuda()
        self.ker_unlbl = torch.exp(-0.5*dist)
        self.ker_unlbl_ang = kernel_linear(X,Y)
        self.unlbl_sig = self.unlbl_sig/max(self.unlbl_sig)
        tmp,_ = self.ker_unlbl_ang.topk(k = self.k_NN,dim=-1)
        tmp = tmp.mean(dim=-1).view(-1,1)/self.unlbl_sig
        
        self.reject_unlabl[tmp>=self.thrsh_ang] = 1.0
        print(self.thrsh_ang,tmp.mean(),tmp.std(),tmp.min(),tmp.max(),sum(self.reject_unlabl))

        print("Unlabelled: stored feature vectors and kernel matrix")
        return

    def generate_mask(self):
        rand_mask_index = random.randint(0,len(self.mask_names)-1)
        input_mask = Image.open(self.mask_names[rand_mask_index])
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        input_mask = transform_input(input_mask).cuda()
        N,H,W = input_mask.shape
        input_mask = input_mask.view(1,N,H,W)
        rand_mask_H = random.randint(0,H-(self.train_batch_size*self.num_M+1)*self.z_height*8)
        H_mask = (self.train_batch_size*self.num_M)*self.z_height*8
        rand_mask_W = random.randint(0,W-(2)*self.z_width*8)
        W_mask = self.z_width*8
        input_mask = input_mask[:,:,rand_mask_H:H_mask+rand_mask_H,rand_mask_W:rand_mask_W+W_mask]
        B = self.num_M*self.train_batch_size
        input_mask = input_mask.permute(0,2,3,1).reshape(B,self.z_height*8,self.z_width*8,N).permute(0,3,1,2)
        return input_mask 

    def gen_featmaps(self,dataloader,net,device):
        
        Mask_uncer = self.generate_mask()
        count =0
        print("Labelled: started storing feature vectors and kernel matrix")
        for batch_id, train_data in enumerate(dataloader):

            input_im, gt, imgid = train_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            
            net.eval()
            B,N,_,_ = input_im.shape
            input_im = input_im.repeat(self.num_M,1,1,1)
            input_im = input_im + 0.1*Mask_uncer
            pred_image,zy_in = net(input_im)
            tensor_mat = zy_in.view(B,self.num_M,self.z_numchnls,self.z_height,self.z_width).mean(dim=1).data#torch.squeeze(zy_in.data)
            tensor_mat = tensor_mat.view(B,self.z_numchnls,self.z_height,self.z_width)
            sigma_uncer = zy_in.view(B,self.num_M,self.z_numchnls,self.z_height,self.z_width).std(dim=1).sum(dim=[-3,-2,-1])
            # saving latent space feature vectors
            for i in range(B):
                if imgid[i] not in self.dict_lbl.keys():
                    self.dict_lbl[imgid[i]] = count
                    count += 1
                tmp_i = self.dict_lbl[imgid[i]]
                self.Fz_lbl[tmp_i,:,:,:] = tensor_mat[i,:,:,:].data
                self.lbl_sig[tmp_i] = sigma_uncer[i].data
                # tensor = torch.squeeze(tensor_mat[i,:,:,:])
        X = self.Fz_lbl.view(-1,self.z_numchnls*self.z_height*self.z_width)
        Y = self.Fz_lbl.view(-1,self.z_numchnls*self.z_height*self.z_width)
        self.var_Fz_lbl = torch.std(self.Fz_lbl,axis=0)
        self.ker_lbl_ang = kernel_linear(X,Y)
        
        # dist = euclidean_distances(X,Y)**2
        dist = torch.from_numpy(euclidean_distances(X.cpu().numpy(),Y.cpu().numpy())).cuda()
        self.ker_lbl = torch.exp(-0.5*dist**2)

        self.lbl_sig = self.lbl_sig/max(self.lbl_sig)
        self.ker_lbl =self.ker_lbl.fill_diagonal_(-1)
        self.ker_lbl_ang = self.ker_lbl_ang.fill_diagonal_(-1)
        #self.ker_lbl_ang = self.ker_lbl_ang.max(axis=1)
        tmp,_ = self.ker_lbl_ang.topk(k = self.k_NN,dim=-1)
        tmp = tmp.mean(dim=-1).view(-1,1)/self.lbl_sig
        self.thrsh_ang = tmp.mean()-1.5*tmp.std()
        # self.thrsh_ang = min(self.thrsh_ang,0.99)
        print(self.thrsh_ang,tmp.mean(),tmp.std(),tmp.min(),tmp.max())
        tmp,_ = self.ker_lbl.topk(k = self.k_NN,dim=-1)
        tmp = tmp.mean(dim=-1).view(-1,1)/self.lbl_sig
        self.thrsh_dist = tmp.mean()-1.5*tmp.std()
        # self.thrsh_dist = min(self.thrsh_dist,0.04)
        # print(self.thrsh_dist,tmp.mean(),tmp.std(),tmp.min(),tmp.max())

        print("Labelled: stored feature vectors and kernel matrix")
        return
    

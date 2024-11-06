import torch
import torch.nn as nn
from layers.Embed import PositionalEmbedding
import torch.nn.functional as F

import torch

def cal_orthogonal_loss(matrix):

    #print(matrix.shape)
    gram_matrix = torch.matmul(matrix.transpose(-2, -1), matrix)  # 
    #print(gram_matrix.shape)  # (batch_size, m, m)


    one_diag = torch.diagonal(gram_matrix, dim1=-2, dim2=-1)  # 
    #print(one_diag.shape)  # (batch_size, m)
    

    two_diag = torch.diag_embed(one_diag)  # 
    #print(two_diag.shape)  # (batch_size, m, m)


    off_diagonal = gram_matrix - two_diag  # 
    #print(off_diagonal.shape)  # (batch_size, m, m)


    loss = torch.norm(off_diagonal, dim=(-2, -1))  # 
    return loss.mean()  # 


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.use_period_norm = configs.use_period_norm
        self.use_orthogonal = configs.use_orthogonal

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.pad_seq_len = 0
        # 4 8 14 30
        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len
        if self.seq_len > self.seg_num_x*self.period_len:
            self.pad_seq_len = (self.seg_num_x+1)*self.period_len - self.seq_len
            self.seg_num_x += 1
            #print(self.pad_seq_len,self.seg_num_x)
        if self.pred_len >  self.seg_num_y*self.period_len:
            self.seg_num_y+=1
        self.ts2basis = nn.Linear(self.seg_num_x,configs.basis_num)
        self.basis2ts = nn.Linear(configs.basis_num,self.seg_num_y)
        #self.trend_linear = nn.Linear(self.period_len,self.period_len)
    def forward(self, x):
        '''
        x: b t c
        out: b t c
        '''
        batch_size = x.shape[0]
        x = x.permute(0,2,1)
        if self.pad_seq_len>0:#、
            t = (self.seg_num_x-1)*self.pred_len
            x = torch.cat([x,x[:,:,t-self.pad_seq_len:t]],dim=-1)
        # b c t-> b c n p ->b c p n -> bc p n
        x = x.reshape(batch_size,self.enc_in,self.seg_num_x,self.period_len)
        x = x.permute(0,1,3,2)
        x = x.reshape(-1,self.period_len,self.seg_num_x)
        if self.use_period_norm:
            period_mean = torch.mean(x,dim=-1,keepdim=True) #bc p 1
            x = x-period_mean

        # bc p n -> bc p n' -> bc p n''
        x_basis = self.ts2basis(x)
        x = self.basis2ts(x_basis)

        if self.use_period_norm:
            x = x+period_mean

        x = x.reshape(batch_size,self.enc_in,self.period_len,self.seg_num_y).permute(0,1,3,2)
        x = x.reshape(batch_size,self.enc_in,-1)
        x = x.permute(0,2,1)
        if self.use_orthogonal:
            orthogonal_loss = cal_orthogonal_loss(x_basis)
            return x[:,:self.pred_len,:],orthogonal_loss
        else:   
            return x[:,:self.pred_len,:]

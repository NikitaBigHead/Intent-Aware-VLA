"""Per-camera causal encoders + visibility-masked attention fusion + shared temporal GRU.
Heads: 4-class action and 30x8x3 residual forecast of the person-camera (webcam) canonical8 pose."""
import torch
from torch import nn

class CamEncoder(nn.Module):
    def __init__(self,d_in,h=64,p=.2):
        super().__init__();self.inp=nn.Sequential(nn.Linear(d_in,h),nn.SiLU(),nn.Dropout(p));self.gru=nn.GRU(h,h,batch_first=True)
    def forward(self,x):return self.gru(self.inp(x))[0]           # causal: step t sees only steps <= t

class AttentionFusion(nn.Module):
    """At each history step, softmax over camera tokens. Logit = MLP([h_c,t ; r_c,t]) ; cameras with r=0
    (nothing observed, no person) are masked. r_c,t = observed fraction of that camera's joints+hands."""
    def __init__(self,h,cams):
        super().__init__();self.emb=nn.Parameter(torch.zeros(cams,h));self.score=nn.Sequential(nn.Linear(h+1,h),nn.Tanh(),nn.Linear(h,1));self.value=nn.Linear(h,h)
    def forward(self,hs,rel):                                        # hs [B,C,T,H], rel [B,C,T]
        hs=hs+self.emb[None,:,None];logit=self.score(torch.cat((hs,rel[...,None]),-1))[...,0]
        mask=rel<=0;allmasked=mask.all(1,keepdim=True);mask=mask&~allmasked
        w=torch.softmax(logit.masked_fill(mask,-1e4),1)
        return (w[...,None]*self.value(hs)).sum(1),w

class MultiCam(nn.Module):
    def __init__(self,dims,h=64,p=.2):
        super().__init__();self.cams=len(dims);self.enc=nn.ModuleList(CamEncoder(d,h,p) for d in dims);self.fuse=AttentionFusion(h,len(dims))
        self.temporal=nn.GRU(h,h,batch_first=True);self.drop=nn.Dropout(p);self.action=nn.Linear(h,4)
        self.motion=nn.Sequential(nn.Linear(h+8*3,128),nn.SiLU(),nn.Linear(128,30*8*3))
    def forward(self,xs,rel,last_pose):
        hs=torch.stack([e(x) for e,x in zip(self.enc,xs)],1);f,w=self.fuse(hs,rel);z=self.drop(self.temporal(f)[0][:,-1])
        delta=self.motion(torch.cat((z,last_pose.flatten(1)),-1)).view(-1,30,8,3)
        return self.action(z),last_pose[:,None]+delta,w

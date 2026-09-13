import sys
from pathlib import Path
import torch
import torch.nn as nn

ROOT=Path(r'C:\DaveLM-CADAVER')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,r'C:\DaveLM-v0.9')
from treatment13_model import Treatment13Model

class BoundedLocalizer(nn.Module):
    def __init__(self, W, b, rho):
        super().__init__()
        self.u=nn.Parameter(((W[0]+W[1])/2).detach().clone())
        self.v=nn.Parameter(((W[0]-W[1])/2).detach().clone())
        self.b_s=nn.Parameter(((b[0]+b[1])/2).detach().clone())
        self.b_r=nn.Parameter(((b[0]-b[1])/2).detach().clone())
        self.register_buffer('rho', torch.tensor(float(rho),dtype=W.dtype))
    def forward(self, hidden):
        S=hidden.matmul(self.u)+self.b_s
        R=self.rho*torch.tanh(hidden.matmul(self.v)+self.b_r)
        return torch.stack((S+R,S-R),dim=-1)

def make_bounded_model(state_dict, rho, device):
    m=Treatment13Model().to(device)
    if 'localizer.scorer.weight' in state_dict:
        m.load_state_dict(state_dict, strict=True)
        old=m.localizer.scorer
        m.localizer=BoundedLocalizer(old.weight,old.bias,rho).to(device)
    else:
        base_state={k:v for k,v in state_dict.items() if not k.startswith('localizer.')}
        missing,unexpected=m.load_state_dict(base_state, strict=False)
        assert set(missing)=={'localizer.scorer.weight','localizer.scorer.bias'} and not unexpected
        dummyW=torch.zeros(2,320,device=device,dtype=state_dict['localizer.u'].dtype)
        dummyb=torch.zeros(2,device=device,dtype=state_dict['localizer.b_s'].dtype)
        loc=BoundedLocalizer(dummyW,dummyb,rho).to(device)
        with torch.no_grad():
            loc.u.copy_(state_dict['localizer.u']); loc.v.copy_(state_dict['localizer.v'])
            loc.b_s.copy_(state_dict['localizer.b_s']); loc.b_r.copy_(state_dict['localizer.b_r'])
            if 'localizer.rho' in state_dict: loc.rho.copy_(state_dict['localizer.rho'])
        m.localizer=loc
    return m

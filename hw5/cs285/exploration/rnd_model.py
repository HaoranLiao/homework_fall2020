from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch

def init_method_1(model):
    try:
        model.weight.data.uniform_()
        model.bias.data.uniform_()
    except:
        pass

def init_method_2(model):
    try:
        model.weight.data.normal_()
        model.bias.data.normal_()
    except:
        pass


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # TODO: Create two neural networks: -------------------
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f
        # WARNING: Make sure you use different types of weight 
        #          initializations for these two functions

        # HINT 1) Check out the method ptu.build_mlp
        # HINT 2) There are two weight init methods defined above

        # self.f = None
        # self.f_hat = None

        self.f = ptu.build_mlp(self.ob_dim, self.output_size, self.n_layers, self.size)
        self.f_hat = ptu.build_mlp(self.ob_dim, self.output_size, self.n_layers, self.size)
        # self.f.apply(init_method_1)
        # self.f_hat.apply(init_method_2)
        init_method_1(self.f[0])
        init_method_2(self.f_hat[0])
        
        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

        self.loss = nn.MSELoss() 
        self.f.to(ptu.device)
        self.f_hat.to(ptu.device)

    def forward(self, ob_no):
        # TODO: Get the prediction error for ob_no --------------------
        # HINT: Remember to detach the output of self.f!
        # error = None
        error = torch.sum(nn.MSELoss(reduction="none")(self.f(ob_no).detach(), self.f_hat(ob_no)), dim=-1)
        return error

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # TODO: Update f_hat using ob_no -----------------
        # loss = None
        ob_no = ptu.from_numpy(ob_no)
        loss = torch.mean(self(ob_no))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
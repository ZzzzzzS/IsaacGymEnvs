from abc import abstractmethod
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from .helper import RLRunningMeanStd

ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'elu': nn.ELU,
    'selu': nn.SELU,
    'swish': nn.SiLU,
    'gelu': nn.GELU,
    'softplus': nn.Softplus,
    'None': nn.Identity
}


class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        # self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    def forward(self, input_obs):
        return self.architecture(input_obs)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

        [torch.nn.init.constant_(module.bias, 0) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class MLP_NO_OUT(nn.Module):
    def __init__(self, input_size, shape, actionvation_fn=nn.ELU):
        super(MLP_NO_OUT, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        # self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [shape[-1]]

    def forward(self, input_obs):
        return self.architecture(input_obs)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

        [torch.nn.init.constant_(module.bias, 0) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class BasicA2CNetwork(nn.Module):
    def __init__(self, obsv_dim, act_dim, value_dim, device, rnn=False, input_norm=False, value_norm=False):
        super().__init__()
        self.obsv_dim = obsv_dim
        self.act_dim = act_dim
        self.value_dim = value_dim
        self.device = device
        self.input_norm = RLRunningMeanStd(
            self.obsv_dim) if input_norm else None
        self.value_norm = RLRunningMeanStd(
            self.value_dim) if value_norm else None
        self.rnn = rnn
        self.rnn_states_dim = None

    def forward(self, obsv, state=None):
        if self.input_norm is not None:
            obsv = self.input_norm(obsv)

        act, value, next_state = self.inference(obsv, state)

        if self.value_norm is not None:
            value = self.value_norm(value, denorm=True)

        return act, value, next_state

    @abstractmethod
    def inference(self, obsv, state=None):
        pass

    @abstractmethod
    def predict(self, obsv, state=None):
        pass

    @abstractmethod
    def action(self, obsv, state=None):
        pass

    @abstractmethod
    def get_default_rnn_states(self):
        pass

    @property
    def obsv_shape(self):
        return self.obsv_dim

    @property
    def act_shape(self):
        return self.act_dim

    @property
    def value_shape(self):
        return self.value_dim

    @property
    def rnn_states_shape(self):
        return self.rnn_states_dim if self.rnn else None

    def is_rnn(self):
        return self.rnn

    def get_value_norm(self):
        return self.value_norm

    @staticmethod
    @abstractmethod
    def create(obsv_dim, act_dim, value_dim, input_norm, value_norm, device, net_config):
        pass


class seperate_symmetry_MLP_A2CNetwork(BasicA2CNetwork):
    def __init__(self, obsv_dim, act_dim, value_dim, hidden_layer, activation_fn=nn.ELU, device="cpu", input_norm=None, value_norm=None):
        super().__init__(obsv_dim, act_dim, value_dim, device, False,
                         input_norm, value_norm)
        self.actor_net = MLP(hidden_layer, activation_fn,
                             self.obsv_dim, self.act_dim)
        self.critic_net = MLP(hidden_layer, activation_fn,
                              self.obsv_dim, self.value_dim)

    def inference(self, obsv, state=None):
        act_mu = self.actor_net(obsv)
        value = self.critic_net(obsv)
        return act_mu, value, None

    @staticmethod
    def create(obsv_dim, act_dim, value_dim, input_norm, value_norm, device, net_config):
        hidden_layer = net_config["mlp"]["units"]
        activation_fn = ACTIVATION_MAP.get(net_config["mlp"]["activation"])
        net = seperate_symmetry_MLP_A2CNetwork(obsv_dim=obsv_dim,
                                               act_dim=act_dim,
                                               value_dim=value_dim,
                                               hidden_layer=hidden_layer,
                                               activation_fn=activation_fn,
                                               device=device,
                                               input_norm=input_norm,
                                               value_norm=value_norm)
        print("network:",net)
        return net

class seperate_symmetry_basic_rnn_A2CNetwork(BasicA2CNetwork):
    """simple rnn net, rnn before mlp, support gru and lstm

    Args:
        BasicA2CNetwork (_type_): _description_
    """
    
    def __init__(self, obsv_dim, act_dim, value_dim,mlp_hidden_layer,mlp_activation,rnn_name,rnn_layers,concat_input,rnn_before,rnn_layer_norm, device, input_norm=False, value_norm=False, rnn_states_dim=None):
        super().__init__(obsv_dim, act_dim, value_dim, device, True, input_norm, value_norm)
        self.concat_input=concat_input
        self.rnn_before=rnn_before
        self.rnn_name=rnn_name
        self.rnn_layers=rnn_layers
        # compute net i/o dim
        if self.rnn_before: #rnn is before mlp
            self.rnn_in_dim=obsv_dim
            self.rnn_out_dim=rnn_states_dim
            self.mlp_in_dim=self.rnn_out_dim
            self.mlp_out_dim=mlp_hidden_layer[-1]
            if self.concat_input:
                self.mlp_in_dim+=obsv_dim
        else:
            self.mlp_in_dim=obsv_dim
            self.mlp_out_dim=mlp_hidden_layer[-1]
            self.rnn_in_dim=self.mlp_out_dim
            self.rnn_out_dim=rnn_states_dim
            if self.concat_input:
                self.rnn_in_dim+=obsv_dim
        
        # build nets
        self.__build_mlp(mlp_hidden_layer,mlp_activation)
        self.__build_rnn(rnn_layer_norm)
    
    def inference(self, obsv:Tensor, state:Tensor):
        
        actor_state=state[:self.D*self.rnn_layers,:,:]
        critic_state=state[self.D*self.rnn_layers:,:,:]
        actor_state=actor_state.contiguous()
        critic_state=critic_state.contiguous()
        
        if self.rnn_before:
            a_out=c_out=obsv.unsqueeze(dim=0)
            a_out,actor_state_next=self.actor_rnn(a_out,actor_state)
            c_out,critic_state_next=self.critic_rnn(c_out,critic_state)
            a_out=a_out.squeeze(dim=0)
            c_out=c_out.squeeze(dim=0)
            a_out=self.actor_rnn_norm(a_out)
            c_out=self.critic_rnn_norm(c_out)
            if self.concat_input:
                a_out=torch.cat((a_out,obsv),dim=-1)
                c_out=torch.cat((c_out,obsv),dim=-1)
            a_out=self.actor_mlp(a_out)
            c_out=self.critic_mlp(c_out)
        else:
            a_out=self.actor_mlp(obsv)
            c_out=self.critic_mlp(obsv)
            
            if self.concat_input:
                a_out=torch.cat((a_out,obsv),dim=-1)
                c_out=torch.cat((c_out,obsv),dim=-1)
            
            a_out=a_out.unsqueeze(dim=0)
            c_out=c_out.unsqueeze(dim=0)
            a_out,actor_state_next=self.actor_rnn(a_out,actor_state)
            c_out,critic_state_next=self.critic_rnn(c_out,critic_state)
            a_out=a_out.squeeze(dim=0)
            c_out=c_out.squeeze(dim=0)
            a_out=self.actor_rnn_norm(a_out)
            c_out=self.critic_rnn_norm(c_out)
            
        next_state=torch.cat([actor_state_next,critic_state_next],dim=0)
        act_mu=self.actor_mu(a_out)
        value=self.value(c_out)
        return act_mu,value,next_state
    
    def get_default_rnn_states(self):
        new_state=torch.zeros(self.rnn_states_shape,dtype=torch.float32,device=self.device)
        return new_state
            
    def __build_mlp(self,mlp_hidden_layers,activation_fn):
        self.actor_mlp=MLP_NO_OUT(self.mlp_in_dim,mlp_hidden_layers,activation_fn)
        self.critic_mlp=MLP_NO_OUT(self.mlp_in_dim,mlp_hidden_layers,activation_fn)
        self.actor_mu=nn.Linear(mlp_hidden_layers[-1],self.act_dim)
        self.value=nn.Linear(mlp_hidden_layers[-1],self.value_dim)
        
    def __build_rnn(self,layer_norm):
        if self.rnn_name=='gru':
            gru_para_dict={
                "input_size":self.rnn_in_dim,
                "hidden_size":self.rnn_out_dim,
                "num_layers":self.rnn_layers
            }
            self.actor_rnn=nn.GRU(**gru_para_dict)
            self.critic_rnn=nn.GRU(**gru_para_dict)
            self.D=1
            self.rnn_states_dim=[2*self.rnn_layers*self.D,1,self.rnn_out_dim]
            
        else:
            raise NotImplementedError() #only support gru currently
        
        self.actor_rnn_norm=nn.LayerNorm(self.rnn_out_dim) if layer_norm else nn.Identity()
        self.critic_rnn_norm=nn.LayerNorm(self.rnn_out_dim) if layer_norm else nn.Identity()
    
    
    @staticmethod
    def create(obsv_dim, act_dim, value_dim, input_norm, value_norm, device, net_config):
        hidden_layer = net_config["mlp"]["units"]
        activation_fn = ACTIVATION_MAP.get(net_config["mlp"]["activation"])
        #rnn:
        #name: lstm
        #units: 256 #128
        #layers: 1
        #before_mlp: False #True
        #concat_input: True
        #layer_norm: False
        name=net_config["rnn"]["name"]
        layers=net_config["rnn"]["layers"]
        concat_input=net_config["rnn"]["concat_input"]
        units=net_config["rnn"]["units"]
        before_mlp=net_config["rnn"].get("before_mlp",True)
        rnn_layer_norm=net_config["rnn"].get("layer_norm",False)
        
        
        net=seperate_symmetry_basic_rnn_A2CNetwork(obsv_dim=obsv_dim,
                                                   act_dim=act_dim,
                                                   value_dim=value_dim,
                                                   mlp_hidden_layer=hidden_layer,
                                                   mlp_activation=activation_fn,
                                                   rnn_name=name,
                                                   rnn_layers=layers,
                                                   concat_input=concat_input,
                                                   rnn_before=before_mlp,
                                                   rnn_layer_norm=rnn_layer_norm,
                                                   device=device,
                                                   input_norm=input_norm,
                                                   value_norm=value_norm,
                                                   rnn_states_dim=units)
        print("network:",net)
        return net
        
    
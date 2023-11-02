from .module import A2CModule_Continuous
from .network import seperate_symmetry_MLP_A2CNetwork
import torch.nn as nn

NET_MAP = {
    'seperate_symmetry_MLP_A2CNetwork': seperate_symmetry_MLP_A2CNetwork
}

MODULE_MAP = {
    'A2CModule_Continuous': A2CModule_Continuous
}

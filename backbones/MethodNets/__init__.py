from .USNID import USNIDModel
from .MCN import MCNModel
from .CC import CCModel
from .SCCL import SCCLModel
from .UMC import UMCModel
from .UMC2 import UMC2Model
from .testmethod import TestmethodModel
from .SMC import SMCModel
methods_map = {
    'usnid': USNIDModel,
    'mcn': MCNModel,
    'cc': CCModel,
    'sccl': SCCLModel,
    'umc': UMCModel,
    'umc2': UMC2Model,
    # 半监督
    'testmethod':TestmethodModel,
    'smc': SMCModel,
}

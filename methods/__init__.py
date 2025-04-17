from .unsupervised.USNID.manager import UnsupUSNIDManager
from .unsupervised.MCN.manager import MCNManager
from .unsupervised.CC.manager import CCManager
from .unsupervised.SCCL.manager import SCCLManager
from .unsupervised.UMC.manager import UMCManager
from .unsupervised.UMC2.manager import UMC2Manager

from .semi_supervised.testmethod.manager import TestManager
from .semi_supervised.SMC.manager import SMCManager
# from .semi_supervised.testmethod.manager import PretrainTestmethod
method_map = {
    # 无监督
    'usnid': UnsupUSNIDManager,
    'mcn': MCNManager,
    'cc': CCManager,
    'sccl': SCCLManager,
    'umc': UMCManager,
    'umc2': UMC2Manager,
    # 半监督
    'testmethod':TestManager,
    'smc': SMCManager,

}

from .MCN import MCN
from .UMC import UMC
from .UMC2 import UMC2
from .BERT_TEXT import BERT_TEXT
multimodal_methods_map = {
    'mcn': MCN,
    'umc': UMC,
    'umc2': UMC2,
    'text': BERT_TEXT,
}
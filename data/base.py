import os
import sys
import logging
import csv
import copy
import random
import numpy as np

from .mm_pre import MMDataset
from .text_pre import get_t_data
from .utils import get_v_a_data
from .__init__ import benchmarks

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

__all__ = ['DataManager']

class DataManager:
    def __init__(self, args):
        """
        在 DataManager 初始化时:
        1) 读取数据集元信息(序列长度/特征维度/全部标签等)并保存到 args 或属性中;
        2) 根据 setting 一次性读取 train/dev/test 样本;
        3) 按无监督/半监督的需求拆分或合并数据集;
        4) 构建对应的 DataLoader。
        """
        self.logger = logging.getLogger(args.logger_name)
        # ---- Step 0: 读取数据集在 benchmarks 中的配置，设置各模态最大长度和维度 ----

        bm = benchmarks[args.dataset]
        max_seq_lengths = bm['max_seq_lengths']
        feat_dims = bm['feat_dims']

        args.text_seq_len, args.video_seq_len, args.audio_seq_len = max_seq_lengths['text'], max_seq_lengths['video'], max_seq_lengths['audio']
        args.text_feat_dim, args.video_feat_dim, args.audio_feat_dim = feat_dims['text'], feat_dims['video'], feat_dims['audio']
        # self.mm_data, self.train_outputs = get_data(args, self.logger)  # NOTE 注释掉防止重复

        ####################### <<START>> lzh:Semi-Supervised part ####################### 
        # ---- Step 1: 准备所有标签、数据路径等基础信息 ----

        self.all_label_list = copy.deepcopy(bm["labels"])
        data_path = os.path.join(args.data_path, args.dataset)
        base_attrs = {
            'data_path': data_path,
            'all_label_list' : self.all_label_list
        }
        # 如半监督，需要先抽取已知标签
        if args.setting == 'semi_supervised':
            self.logger.info('【半监督】进入了半监督的设定中, 随机抽取标签')

            args.n_known_cls = self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
            self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))

            self.logger.info('The number of known intents is %s', self.n_known_cls)
            self.logger.info('Lists of known labels are: %s', str(self.known_label_list))
            base_attrs['known_label_list'] = self.known_label_list
            # 半监督下，最终聚类个数可能与已知标签数不完全一致
            args.num_labels = self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)
        elif args.setting == 'unsupervised':
            args.num_labels = self.num_labels = len(self.all_label_list)

        # ---- Step 2: 一次性 读取 train/dev/test 的原始样本 ----
        train_data = get_examples(args, base_attrs, 'train')  # 根据 setting，可能返回一个 list 或一个 tuple
        dev_data   = get_examples(args, base_attrs, 'dev')
        test_data  = get_examples(args, base_attrs, 'test')

        if args.setting == 'unsupervised':
            self.logger.info('【无监督】进入了无监督的设定中, 加载数据集')
            
            self.train_examples = train_data + dev_data # LZH

            self.train_dataloader, self.train_outputs = get_data(
                self.train_examples, args, self.all_label_list, self.logger, 'train'
            )
            self.logger.info("Number of train samples = %s", str(len(self.train_examples)))

            # self.test_examples = get_examples(args, base_attrs, 'test')
            self.test_examples = test_data
            self.logger.info("Number of testing samples = %s", str(len(self.test_examples)))
            # self.test_dataloader = get_data(self.test_examples, args, base_attrs['all_label_list'], self.logger, 'test')
            self.test_dataloader = get_data(
                self.test_examples, args, self.all_label_list, self.logger, 'test'
            )
            self.eval_dataloader = None

        elif args.setting == 'semi_supervised': 
            # 在半监督场景下，get_examples(..., 'train') 会返回三份: (ori_examples, labeled, unlabeled)
            (self.train_examples,
             self.train_labeled_examples,
             self.train_unlabeled_examples) = train_data
            
            self.logger.info("Number of labeled training samples = %s", str(len(self.train_labeled_examples)))
            self.logger.info("Number of unlabeled training samples = %s", str(len(self.train_unlabeled_examples)))

            # self.eval_examples = get_examples(args, base_attrs, 'dev')
            self.eval_examples = dev_data

            self.logger.info("Number of evaluation samples = %s", str(len(self.eval_examples)))
            # self.test_examples = get_examples(args, base_attrs, 'test')
            self.test_examples = test_data
            self.logger.info("Number of testing samples = %s", str(len(self.test_examples)))
            

            # 分别构建 dataloader
            # 1) labeled & unlabeled 的单独 dataloader
            self.train_labeled_dataloader, self.train_labeled_mmdata = get_data(
                self.train_labeled_examples, args, self.known_label_list, self.logger, 'train_labeled'  # NOTE 这里是train_labeled 为何会有mmdata
            )
            self.train_unlabeled_dataloader, self.train_unlabeled_mmdata = get_data(
                self.train_unlabeled_examples, args, self.all_label_list, self.logger, 'train_unlabeled'
            )
            # 2) 合并后的 dataloader (semi场景可能需要一起训练)
            self.train_dataloader, self.train_out = get_semi_data(
                self.train_labeled_examples,
                self.train_unlabeled_examples,
                self.train_labeled_mmdata,
                self.train_unlabeled_mmdata,
                base_attrs,  # 用于获取路径、label列表等
                args
            )
            # # 3) 构建 eval_dataloader / test_dataloader
            self.eval_dataloader = get_data(
                self.eval_examples, args, self.known_label_list, self.logger, 'dev'
            ) 
            self.test_dataloader = get_data(
                self.test_examples, args, self.all_label_list, self.logger, 'test'
            )
        else:
            # 如果还有其他 setting（例如纯监督 supervised），你可以在此添加自己的逻辑
            # 这里给一个示例：train = train_data, dev = dev_data
            self.train_examples = train_data
            self.eval_examples  = dev_data
            self.test_examples  = test_data

            # 构建 train / eval / test dataloader
            self.train_dataloader, self.train_outputs = get_data(
                self.train_examples, args, self.all_label_list, self.logger, 'train'
            )
            self.eval_dataloader = get_data(
                self.eval_examples, args, self.all_label_list, self.logger, 'dev'
            )
            self.test_dataloader = get_data(
                self.test_examples, args, self.all_label_list, self.logger, 'test'
            )
        
        args.num_train_examples = len(self.train_examples)
        self.logger.info("num_train_examples = %d", args.num_train_examples)

        ####################### <<END>> lzh:Semi-Supervised part ####################### 
####################### <<START>> lzh:Semi-Supervised part ####################### 
  
def get_data(examples, args, label_list, logger, mode):
    
    data_path = os.path.join(args.data_path, args.dataset)
    
    logger.info('data preparation...')
    
    label_ids = get_ids_annotations(args, examples,label_list,mode)
        
    text_data = get_t_data(args, examples)
        
    video_feats_path = os.path.join(data_path, args.video_data_path, args.video_feats_path)
    video_data = get_v_a_data(args, examples, video_feats_path, args.video_seq_len)
    
    audio_feats_path = os.path.join(data_path, args.audio_data_path, args.audio_feats_path)
    audio_data = get_v_a_data(args, examples, audio_feats_path, args.audio_seq_len)
    
    mm_data = MMDataset(label_ids, text_data, video_data, audio_data)

    if mode in ['train','train_labeled','train_unlabeled']:
        dataloader = DataLoader(mm_data, shuffle=False, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)
    if mode == 'test':
        dataloader = DataLoader(mm_data, batch_size = args.test_batch_size, num_workers = args.num_workers, pin_memory = True)
    if mode == 'dev':
        dataloader = DataLoader(mm_data, batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)

    if mode == 'train':
        train_out = {
            'text': text_data,
            'video': video_data,
            'audio': audio_data,
            'label_ids': label_ids,
        }
        
        return dataloader, train_out
    elif mode in ['train_labeled','train_unlabeled']:
        return dataloader,mm_data
    else:
        return dataloader
####################### <<END>> lzh:Semi-Supervised part ####################### 

####################### <<START>> lzh:Semi-Supervised part ####################### 
def get_semi_data(labeled_examples, unlabeled_examples, train_labeled_mmdata,train_unlabeled_mmdata,base_attrs,args):
    data_path = os.path.join(args.data_path, args.dataset)
    semi_examples = labeled_examples + unlabeled_examples
    text_data = get_t_data(args, semi_examples)
        
    video_feats_path = os.path.join(data_path, args.video_data_path, args.video_feats_path)
    video_data = get_v_a_data(args, semi_examples, video_feats_path, args.video_seq_len)
    
    audio_feats_path = os.path.join(data_path, args.audio_data_path, args.audio_feats_path)
    audio_data = get_v_a_data(args, semi_examples, audio_feats_path, args.audio_seq_len)

    labeled_ids = get_ids_annotations(args, labeled_examples,base_attrs['known_label_list'],'train_labeled')
    unlabeled_ids = get_ids_annotations(args, unlabeled_examples,base_attrs['all_label_list'],'train_unlabeled')
    label_ids = labeled_ids + unlabeled_ids
    
    combined_dataset = ConcatDataset([train_labeled_mmdata, train_unlabeled_mmdata])
    dataloader = DataLoader(combined_dataset, shuffle=False, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)

    train_out = {
            'text': text_data,
            'video': video_data,
            'audio': audio_data,
            'label_ids': label_ids,
        }
    return dataloader,train_out

def get_examples(args, base_attrs, mode):

    processor = DatasetProcessor(args)
    ori_examples = processor.get_examples(args,base_attrs['data_path'], mode)
    
    if args.setting == 'unsupervised':
        
        return ori_examples

    elif args.setting == 'semi_supervised':

        if mode == 'train':
            
            train_labels = np.array([example.label for example in ori_examples])
            train_labeled_ids = []
            for label in base_attrs['known_label_list']:
                num = round(len(train_labels[train_labels == label]) * args.labeled_ratio)
                pos = list(np.where(train_labels == label)[0])
                train_labeled_ids.extend(random.sample(pos, num))

            labeled_examples, unlabeled_examples = [], []
            for idx, example in enumerate(ori_examples):
                if idx in train_labeled_ids:
                    labeled_examples.append(example)
                else:
                    unlabeled_examples.append(example)

            return ori_examples, labeled_examples, unlabeled_examples

        elif mode == 'dev':
            if args.merge_dev: # 如果需要将 dev 和 train 合并，那么和train做相同的处理
                dev_labels = np.array([example.label for example in ori_examples])
                dev_labeled_ids = []
                for label in base_attrs['known_label_list']:
                    num = round(len(dev_labels[dev_labels == label]) * args.labeled_ratio)
                    pos = list(np.where(dev_labels == label)[0])
                    dev_labeled_ids.extend(random.sample(pos, num))

                labeled_examples, unlabeled_examples = [], []
                for idx, example in enumerate(ori_examples):
                    if idx in dev_labeled_ids:
                        labeled_examples.append(example)
                    else:
                        unlabeled_examples.append(example)

                return ori_examples, labeled_examples, unlabeled_examples

            else:
                examples = []
                for example in ori_examples:
                    if (example.label in base_attrs['known_label_list']):
                        examples.append(example)
                
                return examples
        
        elif mode == 'test':
            return ori_examples

def get_ids_annotations(args, examples,label_list,mode):

    if mode == 'train_unlabeled':   # NOTE 这里是mask掉
        label_ids = [-1] * len(examples)

    else:
        label_map = {}
        label_ids = []
        for i, label in enumerate(label_list): # NOTE 映射
            label_map[label] = i

        for example in examples:
            label_id = label_map[example.label] # NOTE 具体准换样本的标签
            label_ids.append(label_id)
    
    return label_ids

####################### <<END>> lzh:Semi-Supervised part ####################### 


####################### <<START>> lzh:Semi-Supervised part ####################### 

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, index=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.index = index

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class DatasetProcessor(DataProcessor):

    def __init__(self, args):
        super(DatasetProcessor).__init__()
        
        if args.dataset in ['MIntRec']:
            self.select_id = 3
            self.label_id = 4
        elif args.dataset in ['clinc', 'clinc-small', 'snips', 'atis']:
            self.select_id = 0
        elif args.dataset in ['L-MIntRec']:
            self.select_id = 5
        elif args.dataset in ['MELD-DA']:
            self.select_id = 2
            self.label_id = 3
        elif args.dataset in ['IEMOCAP-DA']:
            self.select_id = 1
            self.label_id = 2
        
    def get_examples(self, args, data_dir, mode):
        
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train",args)
        elif mode == 'dev':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train",args)
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test",args)
        elif mode == 'all':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "all.tsv")), "all",args)

    def _create_examples(self, lines, set_type,args):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, i)
            text_a = line[self.select_id]
            label = line[self.label_id] 

            if args.dataset in ['MIntRec']:
                index = '_'.join([line[0], line[1], line[2]])
                
            elif args.dataset in ['MELD-DA']:
                index = '_'.join([line[0], line[1]])
            
            elif args.dataset in ['IEMOCAP-DA']:
                index = line[0]
                

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None,label=label,index=index))
        return examples
####################### <<END>> lzh:Semi-Supervised part ####################### 

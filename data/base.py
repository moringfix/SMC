import os
import logging
import csv
import copy
import random
from .mm_pre import MMDataset
from .text_pre import get_t_data
from .utils import get_v_a_data
from .__init__ import benchmarks
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

__all__ = ['DataManager']

class DataManager:
    
    def __init__(self, args):
        
        self.logger = logging.getLogger(args.logger_name)
        bm = benchmarks[args.dataset]
        max_seq_lengths, feat_dims = bm['max_seq_lengths'], bm['feat_dims']
        args.text_seq_len, args.video_seq_len, args.audio_seq_len = max_seq_lengths['text'], max_seq_lengths['video'], max_seq_lengths['audio']
        args.text_feat_dim, args.video_feat_dim, args.audio_feat_dim = feat_dims['text'], feat_dims['video'], feat_dims['audio']
        self.mm_data, self.train_outputs = get_data(args, self.logger) 

        ####################### <<START>> lzh:Semi-Supervised part ####################### 
        self.all_label_list = copy.deepcopy(bm["labels"])
        data_path = os.path.join(args.data_path, args.dataset)
        base_attrs = {
            'data_path': data_path,
            'all_label_list' : self.all_label_list
        }
        


        if args.setting == 'semi_supervised':
            
            args.n_known_cls = self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
            self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))

            self.logger.info('The number of known intents is %s', self.n_known_cls)
            self.logger.info('Lists of known labels are: %s', str(self.known_label_list))
            base_attrs['known_label_list'] = self.known_label_list
            args.num_labels = self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)

        if args.setting == 'unsupervised':

            self.train_examples = get_examples(args, base_attrs, 'train')
            self.eval_examples = get_examples(args, base_attrs, 'dev')
            # print('111111111',self.train_examples)
            # print('222222222',self.eval_examples)

            self.train_examples = self.train_examples + self.eval_examples
            # args.num_train_examples = len(self.train_examples)
            self.train_dataloader,self.train_outputs = get_data(self.train_examples, args, base_attrs['all_label_list'], self.logger, 'train')
            self.logger.info("Number of train samples = %s", str(len(self.train_examples)))

            self.test_examples = get_examples(args, base_attrs, 'test')
            self.logger.info("Number of testing samples = %s", str(len(self.test_examples)))
            self.test_dataloader = get_data(self.test_examples, args, base_attrs['all_label_list'], self.logger, 'test')
            
 
        elif args.setting == 'semi_supervised': 

            self.train_examples, self.train_labeled_examples, self.train_unlabeled_examples  = get_examples(args, base_attrs, 'train')
            self.logger.info("Number of labeled training samples = %s", str(len(self.train_labeled_examples)))
            self.logger.info("Number of unlabeled training samples = %s", str(len(self.train_unlabeled_examples)))
            self.eval_examples = get_examples(args, base_attrs, 'dev')
            self.logger.info("Number of evaluation samples = %s", str(len(self.eval_examples)))
            self.test_examples = get_examples(args, base_attrs, 'test')
            self.logger.info("Number of testing samples = %s", str(len(self.test_examples)))
            
            self.train_labeled_dataloader,self.train_labeled_mmdata = get_data(self.train_labeled_examples, args, base_attrs['known_label_list'], self.logger,'train_labeled')
            self.train_unlabeled_dataloader,self.train_unlabeled_mmdata = get_data(self.train_unlabeled_examples, args, base_attrs['all_label_list'], self.logger,'train_unlabeled')
            self.train_dataloader,self.train_out = get_semi_data(self.train_labeled_examples, self.train_unlabeled_examples, self.train_labeled_mmdata, self.train_unlabeled_mmdata, base_attrs, args)
            self.eval_dataloader = get_data(self.eval_examples, args, base_attrs['known_label_list'], self.logger,'dev')
            self.test_dataloader = get_data(self.test_examples, args, base_attrs['all_label_list'], self.logger,'test')

        args.num_train_examples = len(self.train_examples)
        ####################### <<END>> lzh:Semi-Supervised part ####################### 


def get_data(args, logger):
    
    data_path = os.path.join(args.data_path, args.dataset)
    bm = benchmarks[args.dataset]
    
    label_list = copy.deepcopy(bm["labels"])
    logger.info('Lists of intent labels are: %s', str(label_list))  
      
    args.num_labels = len(label_list)  
    
    logger.info('data preparation...')
    
    train_data_index, train_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path, 'train.tsv'))
    dev_data_index, dev_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path, 'dev.tsv'))
    
    train_data_index = train_data_index + dev_data_index
    train_label_ids = train_label_ids + dev_label_ids

    test_data_index, test_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path, 'test.tsv'))
    args.num_train_examples = len(train_data_index)
    
    data_args = {
        'data_path': data_path,
        'train_data_index': train_data_index,
        'test_data_index': test_data_index,
    }
        
    text_data = get_t_data(args, data_args)
        
    video_feats_path = os.path.join(data_args['data_path'], args.video_data_path, args.video_feats_path)
    video_data = get_v_a_data(data_args, video_feats_path, args.video_seq_len)
    
    audio_feats_path = os.path.join(data_args['data_path'], args.audio_data_path, args.audio_feats_path)
    print('###################### Audio feats path:' ,audio_feats_path)
    audio_data = get_v_a_data(data_args, audio_feats_path, args.audio_seq_len)
    
    mm_train_data = MMDataset(train_label_ids, text_data['train'], video_data['train'], audio_data['train'])
    mm_test_data = MMDataset(test_label_ids, text_data['test'], video_data['test'], audio_data['test'])

    mm_data = {'train': mm_train_data, 'test': mm_test_data}
    
    train_outputs = {
        'text': text_data['train'],
        'video': video_data['train'],
        'audio': audio_data['train'],
        'label_ids': train_label_ids,
    }
    
    return mm_data, train_outputs
                 

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

def get_ids_annotations(args, examples,label_list,mode):

    if mode == 'train_unlabeled':
        label_ids = [-1] * len(examples)

    else:
        label_map = {}
        label_ids = []
        for i, label in enumerate(label_list):
            label_map[label] = i

        for example in examples:
            label_id = label_map[example.label]
            label_ids.append(label_id)
    
    return label_ids

####################### <<END>> lzh:Semi-Supervised part ####################### 






def get_indexes_annotations(args, bm, label_list, read_file_path):

    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    with open(read_file_path, 'r') as f:

        data = csv.reader(f, delimiter="\t")
        indexes = []
        label_ids = []

        for i, line in enumerate(data):
            if i == 0:
                continue
            
            if args.dataset in ['MIntRec']:
                index = '_'.join([line[0], line[1], line[2]])
                indexes.append(index)
                
                label_id = label_map[line[4]]
            
            elif args.dataset in ['MELD-DA']:
                label_id = label_map[line[3]]
                
                index = '_'.join([line[0], line[1]])
                indexes.append(index)
            
            elif args.dataset in ['IEMOCAP-DA']:
                label_id = label_map[line[2]]
                index = line[0]
                indexes.append(index)
            
            label_ids.append(label_id)
    
    return indexes, label_ids
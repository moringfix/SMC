import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
import time 
import copy

from sklearn.cluster import KMeans
from tqdm import trange, tqdm
from losses import loss_map
from utils.functions import save_model, restore_model, set_torch_seed
from transformers import BertTokenizer
from sklearn.metrics import silhouette_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

from backbones.base import freeze_bert_parameters
from sklearn.neighbors import NearestNeighbors, KDTree
from utils.neighbor_dataset import NeighborsDataset
from torch.utils.data import DataLoader

from utils.metrics import clustering_score
from .pretrain import PretrainTestmethod

# from data.utils import get_dataloader
from .utils import *

class TestManager:
    def __init__(self, args, data, model):

        pretrain_manager = PretrainTestmethod(args, data, model)

        set_torch_seed(args.seed)

        self.logger = logging.getLogger(args.logger_name)
        self.device, self.model = model.device, model.model

        self.train_labeled_dataloader = data.train_labeled_dataloader
        self.train_unlabeled_dataloader = data.train_unlabeled_dataloader
        self.train_dataloader = data.train_dataloader
        self.train_out = data.train_out
        self.eval_dataloader = data.eval_dataloader
        self.test_dataloader = data.test_dataloader
        self.train_text_data, self.train_video_data, self.train_audio_data = \
            self.train_out['text'], self.train_out['video'], self.train_out['audio']

        self.criterion = loss_map['CrossEntropyLoss']
        self.centroids = None

        self.tokenizer = BertTokenizer.from_pretrained(args.text_pretrained_model, do_lower_case=True)    
        self.generator = view_generator(self.tokenizer, args)

        if args.pretrain:
            self.pretrained_model = pretrain_manager.model

            self.num_labels = args.num_labels = data.num_labels
            self.load_pretrained_model(self.pretrained_model)
            
        else:
            self.num_labels = args.num_labels = data.num_labels
            self.pretrained_model = restore_model(pretrain_manager.model, os.path.join(args.model_output_path, 'pretrain'), self.device)
    
        if args.train:
            # args.num_train_epochs = (1 - args.thres) / args.delta
            self.optimizer, self.scheduler = set_optimizer(args, self.model, args.lr)

            if args.freeze_train_bert_parameters:
                self.logger.info('Freeze all parameters but the last layer for efficiency')
                self.model = freeze_bert_parameters(self.model, args.multimodal_method)
            
            self.load_pretrained_model(self.pretrained_model)
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)   

    def _train(self, args): 

        best_model = None
        wait = 0
        best_eval_score = 0 

        self.model.to(self.device)

        for epoch in trange(int(args.num_train_epochs), desc='Epoch'):

            outputs = self._get_outputs(args, mode = 'train', return_feats = True)
            feats = outputs['feats']
            y_true = outputs['y_true']

            km = KMeans(n_clusters = self.num_labels, random_state=args.seed).fit(feats)
            eval_score = silhouette_score(feats, km.labels_)

            if epoch > 0:
                
                eval_results = {
                    'train_loss': tr_loss,
                    'cluster_silhouette_score': eval_score,
                    'best_cluster_silhouette_score': best_eval_score,   
                }

                self.logger.info("***** Epoch: %s: Eval results *****", str(epoch))
                for key in sorted(eval_results.keys()):
                    self.logger.info("  %s = %s", key, str(round(eval_results[key], 4)))

            if eval_score > best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_eval_score = eval_score
            elif eval_score > 0:
                wait += 1
                if wait >= args.wait_patient:
                    break 
            
            # pseudo_labels = torch.tensor(km.labels_ , dtype=torch.long).to(self.device)
            pseudo_labels = self.alignment(km, args)
            pseudo_train_dataloader = self.update_pseudo_labels(pseudo_labels, args)

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()

            for batch in tqdm(pseudo_train_dataloader, desc="Training(All)"):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):
                    _,mlp_output = self.model(text_feats, video_feats, audio_feats, mode='train-mm')

                    loss = self.criterion(mlp_output,label_ids)
                    self.optimizer.zero_grad()
                    loss.backward()

                    tr_loss += loss.item()
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.scheduler.step()

            tr_loss = tr_loss / nb_tr_steps
        
        self.model = best_model

        if args.save_model:
            save_model(self.model, args.model_output_path)

    def _test(self, args):

        outputs = self._get_outputs(args, mode = 'test', return_feats = True)
        feats = outputs['feats']
        y_true = outputs['y_true']
        km = KMeans(n_clusters = self.num_labels, random_state=args.seed).fit(feats)
        y_pred = km.labels_

        test_results = clustering_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        self.logger.info
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))
        self.logger.info("***** Test results *****")
        
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred
        
        if args.cluster_num_factor > 1:
            test_results['estimate_k'] = args.num_labels

        return test_results

    def _get_outputs(self, args, mode, return_feats = False):
        
        if mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Get Outputs"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):
               
                features, _ = self.model(text_feats, video_feats, audio_feats, mode='train-mm')
                
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, features))
        
        feats = total_features.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        
        if return_feats:
            outputs = {
                'y_true': y_true,
                'feats': feats
            }

        else:
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)
            y_pred = total_preds.cpu().numpy()
            
            y_logits = total_logits.cpu().numpy()
            
            outputs = {
                'y_true': y_true,
                'y_pred': y_pred,
                'logits': y_logits,
                'feats': feats
            }

        return outputs

    def alignment(self, km, args):

        if self.centroids is not None:

            old_centroids = self.centroids.cpu().numpy()
            new_centroids = km.cluster_centers_
            
            DistanceMatrix = np.linalg.norm(old_centroids[:,np.newaxis,:]-new_centroids[np.newaxis,:,:],axis=2) 
            row_ind, col_ind = linear_sum_assignment(DistanceMatrix)
            
            new_centroids = torch.tensor(new_centroids).to(self.device)
            self.centroids = torch.empty(self.num_labels ,args.feat_dim).to(self.device)
            
            alignment_labels = list(col_ind)
            for i in range(self.num_labels):
                label = alignment_labels[i]
                self.centroids[i] = new_centroids[label]
                
            pseudo2label = {label:i for i,label in enumerate(alignment_labels)}
            pseudo_labels = np.array([pseudo2label[label] for label in km.labels_])

        else:
            self.centroids = torch.tensor(km.cluster_centers_).to(self.device)        
            pseudo_labels = km.labels_ 

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)
        
        return pseudo_labels

    def load_pretrained_model(self, pretrained_model):
        
        pretrained_dict = pretrained_model.state_dict()
        mlp_params = ['method_model.mlp_head_train.2.weight', 'method_model.mlp_head_train.2.bias','method_model.train_head.2.weight','method_model.train_head.2.bias','method_model.pretrain_head.2.weight','method_model.pretrain_head.2.bias']
  
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in mlp_params}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def update_pseudo_labels(self, pseudo_labels, args):
        train_data = MMDataset(pseudo_labels, self.train_text_data, self.train_video_data, self.train_audio_data)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = args.train_batch_size)
        return train_dataloader
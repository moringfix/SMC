from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
import os
import logging
import copy

from tqdm import trange, tqdm
from transformers import BertTokenizer
from losses import loss_map
from utils.functions import save_model, restore_model
from .utils import * #set_optimizer, view_generator, get_pseudo_dataloader
from utils.metrics import clustering_score
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.metrics import accuracy_score

from backbones.base import freeze_bert_parameters


class PretrainTestmethod:
    def __init__(self, args, data, model):
        self.logger = logging.getLogger(args.logger_name)

        self.num_labels = args.num_labels 
        self.n_known_cls = data.n_known_cls
        
        self.device, self.model = model.device, model.model
        if args.freeze_pretrain_bert_parameters:
            self.logger.info('Freeze all parameters but the last layer for efficiency')
            self.model = freeze_bert_parameters(self.model, args.multimodal_method)

        self.optimizer, self.scheduler = set_optimizer(args, self.model, args.lr_pre)

        self.train_labeled_dataloader = data.train_labeled_dataloader
        self.train_unlabeled_dataloader = data.train_unlabeled_dataloader
        self.train_dataloader = data.train_dataloader
        self.train_out = data.train_out
        self.eval_dataloader = data.eval_dataloader
        self.test_dataloader = data.test_dataloader

        self.contrast_criterion = loss_map['CrossEntropyLoss']

        self.tokenizer = BertTokenizer.from_pretrained(args.text_pretrained_model, do_lower_case=True)    
        self.generator = view_generator(self.tokenizer, args)

        if args.pretrain:
            
            self.logger.info('Pre-training start...')
            self._train(args)
            self.logger.info('Pre-training finished...')
            
        else:
            self.model = restore_model(self.model, os.path.join(args.model_output_path, 'pretrain'), self.device)

        self.model.to(torch.device('cpu'))
        torch.cuda.empty_cache()

    def _train(self, args):

        wait = 0
        best_model = None
        best_eval_score = 0

        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):  

            self.model.train()
            tr_loss, nb_tr_steps = 0, 0

            for batch in tqdm(self.train_labeled_dataloader, desc = "Iteration"):
                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):
                    _,mlp_output = self.model(text_feats, video_feats, audio_feats, mode='pretrain-mm')

                    loss = self.contrast_criterion(mlp_output,label_ids)
                    self.optimizer.zero_grad()
                    loss.backward()

                    tr_loss += loss.item()
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.scheduler.step()

            loss = tr_loss / nb_tr_steps
            output = self._get_outputs(args, mode = 'eval')
            y_true = output['y_true']
            y_pred = output['y_pred']
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_score':best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score > best_eval_score:
                
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_eval_score = eval_score

            elif eval_score > 0:

                wait += 1
                if wait >= args.wait_patient:
                    break
                
        self.model = best_model

        if args.save_model:
            pretrained_model_dir = os.path.join(args.model_output_path, 'pretrain')
            if not os.path.exists(pretrained_model_dir):
                os.makedirs(pretrained_model_dir)
            save_model(self.model, pretrained_model_dir)

    def _get_outputs(self, args, mode, return_feats = False):
        
        if mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader
        elif mode == 'train_labeled':
            dataloader = self.train_labeled_dataloader
        elif mode == 'eval':
            dataloader = self.eval_dataloader

        self.model.eval()
        self.model.to(self.device)

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, self.n_known_cls)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Get Outputs"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):
               
                features, logits = self.model(text_feats, video_feats, audio_feats, mode='pretrain-mm')
                
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, features))
                total_logits = torch.cat((total_logits, logits))
        
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

    # def _test(self, args):

    #     outputs = self._get_outputs(args, mode = 'test', return_feats = True)
    #     feats = outputs['feats']
    #     y_true = outputs['y_true']
    #     km = KMeans(n_clusters = self.num_labels, random_state=args.seed).fit(feats)
    #     y_pred = km.labels_

    #     test_results = clustering_score(y_true, y_pred)
    #     cm = confusion_matrix(y_true, y_pred)
        
    #     self.logger.info
    #     self.logger.info("***** Test: Confusion Matrix *****")
    #     self.logger.info("%s", str(cm))
    #     self.logger.info("***** Test results *****")
        
    #     for key in sorted(test_results.keys()):
    #         self.logger.info("  %s = %s", key, str(test_results[key]))

    #     test_results['y_true'] = y_true
    #     test_results['y_pred'] = y_pred
        
    #     if args.cluster_num_factor > 1:
    #         test_results['estimate_k'] = args.num_labels

    #     return test_results
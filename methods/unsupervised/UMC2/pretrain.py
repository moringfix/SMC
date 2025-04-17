import torch
import torch.nn.functional as F
import os
import logging

from tqdm import trange, tqdm
from transformers import BertTokenizer
from losses import loss_map
from utils.functions import save_model, restore_model
from .utils import * #set_optimizer, view_generator, get_pseudo_dataloader

from backbones.base import freeze_bert_parameters
import copy, logging, os, torch, torch.nn.functional as F
from tqdm import trange, tqdm
from utils.functions import EarlyStopping
class PretrainUMC2Manager:
    
    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)
        
        self.device, self.model = model.device, model.model
        if args.freeze_pretrain_bert_parameters:
            self.logger.info('Freeze all parameters but the last layer for efficiency')
            self.model = freeze_bert_parameters(self.model, args.multimodal_method)

        self.optimizer, self.scheduler = set_optimizer(args, self.model, args.lr_pre)

        self.train_outputs = data.train_outputs
        
        self.contrast_criterion = loss_map['SupConLoss']
        # self.contrast_criterion = loss_map['SupConLossWeighted'] # NOTE: 这里可以选择使用加权的对比损失函数
        
        print("########################### pretrained_bert_model: ", args.pretrained_bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model, do_lower_case=True)    
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
        
        pseudo_data, pseudo_dataloader = get_pseudo_dataloader(args, self.train_outputs, mode='pretrain')
        early_stopper = EarlyStopping(args, delta=1e-3)

        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            
            self.model.train()
            tr_loss, nb_tr_steps = 0, 0

            for batch in tqdm(pseudo_dataloader, desc = "Iteration"):
                
                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                
                with torch.set_grad_enabled(True):

                    mlp_output_a = self.model(text_feats, torch.zeros_like(video_feats).to(self.device), audio_feats, mode='pretrain-mm')
                    mlp_output_b = self.model(text_feats, video_feats, torch.zeros_like(audio_feats).to(self.device), mode='pretrain-mm')
                    mlp_output_c = self.model(text_feats, video_feats, audio_feats, mode='pretrain-mm')

                    norm_mlp_output_a = F.normalize(mlp_output_a)
                    norm_mlp_output_b = F.normalize(mlp_output_b)
                    norm_mlp_output_c = F.normalize(mlp_output_c)

                    contrastive_logits = torch.cat((norm_mlp_output_a.unsqueeze(1), norm_mlp_output_b.unsqueeze(1), norm_mlp_output_c.unsqueeze(1)), dim = 1) # NOTE 这里dim是1是因为这是视角的维度，0是batch_size的维度
                    loss_contrast_mm = self.contrast_criterion(contrastive_logits, temperature = args.pretrain_temperature, device = self.device)
                    loss = loss_contrast_mm
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    if args.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    tr_loss += loss.item()
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.scheduler.step()

            # # ---------- ⑤ 可选：保存 ----------
            # if args.save_model:
            #     save_dir = os.path.join(args.model_output_path, 'pretrain')
            #     os.makedirs(save_dir, exist_ok=True)
            #     save_model(self.model, save_dir)

            loss = tr_loss / nb_tr_steps
            eval_results = {
                'train_loss': loss,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))

            # ---------- ② 本 epoch 监控值 ----------
            epoch_loss = loss      # ← score
            self.logger.info(f"Epoch {epoch+1} | train_loss = {epoch_loss:.6f}")

            # ---------- ③ Early‑Stop 检查 ----------
            early_stopper(epoch_loss, self.model)
            if early_stopper.early_stop:
                self.logger.info("Early stopping triggered. "
                                f"Best {early_stopper.monitor}: {early_stopper.best_score:.6f}")
                break

        # ---------- ④ 把最好模型取回来 ----------
        if early_stopper.best_model is not None:
            self.model = early_stopper.best_model

        if args.save_model:
            pretrained_model_dir = os.path.join(args.model_output_path, 'pretrain')
            if not os.path.exists(pretrained_model_dir):
                os.makedirs(pretrained_model_dir)
            save_model(self.model, pretrained_model_dir)

import torch.nn as nn

class TestmethodModel(nn.Module):

    def __init__(self, args, backbone):

        super(TestmethodModel, self).__init__()
        self.backbone = backbone
        activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}
        args.feat_dim = args.base_dim
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.train_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(args.hidden_dropout_prob),
            nn.Linear(args.base_dim, args.num_labels), 
        )

        self.pretrain_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(args.hidden_dropout_prob),
            nn.Linear(args.base_dim, args.n_known_cls), 
        )
        
    def freeze_heads(self, mode):
        if mode == 'pretrain-mm':
            for param in self.pretrain_head.parameters():
                param.requires_grad = False
        elif mode == 'train-mm':
            for param in self.train_head.parameters():
                param.requires_grad = False

    def forward(self, text, video, audio, mode='train'):
        
        if mode == 'pretrain-mm':
            features = self.backbone(text, video, audio, mode='features')
            mlp_output = self.pretrain_head(features)
            return features, mlp_output

        elif mode == 'train-mm':
            features = self.backbone(text, video, audio, mode='features')
            mlp_output = self.train_head(features)

            return features, mlp_output
                
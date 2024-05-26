from trainer.trainer import Trainer

import torch
import torch.nn as nn

class WeightedTrainer(Trainer):
    def _init_model(self, nchannel, nclass):
        super()._init_model(nchannel, nclass)

        self.weights = nn.Parameter(torch.ones(4, dtype=torch.float).cuda()/4, requires_grad=True)
        train_params = [{
            'params': self.weights, 'lr': self.args.optim.lr
        }]

        # Define Optimizer
        if self.args.optim.name == 'sgd':
            self.optimizer = torch.optim.SGD(
                train_params, momentum=self.args.optim.momentum, weight_decay=self.args.optim.weight_decay)
        elif self.args.optim.name == 'adam':
            self.optimizer = torch.optim.Adam(
                train_params, weight_decay=self.args.optim.weight_decay)
        
    
    def training(self, epoch):
        super().training(epoch)
        print(self.weights)
            
    def forward_batch(self, image, target):
        # with torch.no_grad():
        output = self.model(image)

        loss = self.criterion(output, target, self.weights)

        return output, loss
    

    def predict(self, image, channel=-1):
        return self.model(x=image, channel=channel, weights=self.weights)
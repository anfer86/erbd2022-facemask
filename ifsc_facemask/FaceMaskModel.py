from audioop import avg
from turtle import forward
import pytorch_lightning as pl
import torchvision.models as models
from torch import nn
import torch.nn.functional as F
from torch.optim import optimizer
import torch

from ifsc_facemask.utils import initialize_model

class FaceMaskModel(pl.LightningModule):

    def __init__(self, model_name = "squeezenet", num_classes = None, feature_extract = True):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.feature_extract = feature_extract
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()

        # init a pretrained model
        self.model, self.input_size = initialize_model(
            model_name=model_name,
            num_classes=num_classes, 
            feature_extract=feature_extract, 
            use_pretrained=True
        )        

    def configure_optimizers(self):
        params_to_update = self.model.parameters()
        if self.feature_extract:
            params_to_update = []
            for name,param in self.model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in self.model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        return torch.optim.Adam(params_to_update, lr=1e-3)       

    def forward(self, x, mode="train"):        
        return self.model(x)
        
    def training_step(self, batch, batch_idx):        
        X, y = batch
        # (1) Passar os dados pela rede neural (forward)        
        # (2) Calcular o erro da saída da rede com a classe das instâncias (loss)
        if self.model_name == "inception":
            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
            # (1) Passar os dados pela rede neural (forward)            
            outputs, aux_outputs = self(X)
            # (2) Calcular o erro da saída da rede com a classe das instâncias (loss)                    
            loss1 = self.criterion(outputs, y)
            loss2 = self.criterion(aux_outputs, y)            
            loss = loss1 + 0.4*loss2
        else:
            # (1) Passar os dados pela rede neural (forward)
            outputs = self(X)
            # (2) Calcular o erro da saída da rede com a classe das instâncias (loss)
            loss = self.criterion(outputs, y)    
        
        return self.step_stats(loss, outputs, y, mode="train")

    def validation_step(self, batch, batch_idx):        
        X, y = batch
        # (1) Passar os dados pela rede neural (forward)
        outputs = self.forward(X)
        # (2) Calcular o erro da saída da rede com a classe das instâncias (loss)
        loss = self.criterion(outputs, y)

        return self.step_stats(loss, outputs, y, mode="val")
    
    def test_step(self, batch, batch_idx):        
        X, y = batch
        # (1) Passar os dados pela rede neural (forward)
        outputs = self.forward(X)
        # (2) Calcular o erro da saída da rede com a classe das instâncias (loss)
        loss = self.criterion(outputs, y)

        return self.step_stats(loss, outputs, y, mode="test")

    def step_stats(self, loss, outputs, y_true, mode="train"):
        _, y_pred = torch.max(outputs, 1)
        n_correct = torch.sum(y_true == y_pred).item()
        n = len(y_true)
        acc = n_correct/n

        self.log('{}_loss'.format(mode), loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('{}_acc'.format(mode), acc, prog_bar=True, on_step=False, on_epoch=True)        
        
        return { "loss" : loss, "acc" : acc, "n_correct" : n_correct, "n" : n}         
    
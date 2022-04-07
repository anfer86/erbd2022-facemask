# Imports básicos
import numpy as np
from numpy import random
import pandas as pd
import math
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')
random.RandomState(1)

# Imports scikit-learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# Imports pytroch
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF
from collections import Counter

def get_transforms(type="train", input_size=224):
      
      transform_dict = {
            'train' : transforms.Compose([
            transforms.RandomResizedCrop(input_size),        
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, hue=0.05),                                            
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                                    
            ]),
            'test' : transforms.Compose([
                  transforms.Resize(input_size),        
                  transforms.ToTensor(),
                  #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
      }

      return transform_dict.get(type, None)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
  
    elif model_name == "vgg16":
        """ VGG16
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "mobilenet_v2":
        """ 
        """
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name =="mobilenet_v3_small":
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[3] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name =="mobilenet_v3_large":
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[3] = nn.Linear(num_ftrs,num_classes)
        input_size = 224        

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size    



def train_epoch(model, trainLoader, optimizer, criterion, device="cpu", is_inception=False):
    model.train()    
    losses = []
    losses_sum = 0
    correct_sum = 0
    total = 0
    pbar = tqdm(trainLoader)
    for X, y in pbar:        
        X, y = X.to(device), y.to(device)        
        optimizer.zero_grad()
        if is_inception:
            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
            # (1) Passar os dados pela rede neural (forward)        
            outputs, aux_outputs = model(X)
            # (2) Calcular o erro da saída da rede com a classe das instâncias (loss)                    
            loss1 = criterion(outputs, y)
            loss2 = criterion(aux_outputs, y)            
            loss = loss1 + 0.4*loss2
        else:
            # (1) Passar os dados pela rede neural (forward)
            outputs = model(X)
            # (2) Calcular o erro da saída da rede com a classe das instâncias (loss)
            loss = criterion(outputs, y)
            _, y_pred = torch.max(outputs, 1)              
                
        # (3) Usar o erro para calcular quanto cada peso (wi) contribuiu com esse erro (backward)
        loss.backward()
        # (4) Ataulizar os pesos da rede neural
        optimizer.step()        
        # (5) Calcular estatísticas do batch
        losses.append(loss.item())
        losses_sum += loss.item()        
        correct_sum += (torch.max(outputs, 1)[1] == y).sum().cpu().data.numpy()
        total += len(y)
        measures = {'train_loss': losses_sum/len(losses), 'train_acc': correct_sum/total }
        pbar.set_postfix(measures)

    model.eval()
    return np.mean(losses)

def eval_model(model, loader, device="cpu", is_inception=False):
      measures = []
      total = 0
      correct = 0
      for X, y in loader:                
            X, y = X.to(device), y.to(device)
            if is_inception:                        
                  output, aux_outputs = model(X)
            else:
                  output = model(X)
            _, y_pred = torch.max(output, 1)                                 
            total += len(y)
            correct += (y_pred == y).sum().cpu().data.numpy()
      measures = {'acc' : correct/total}
      return measures

def train_and_evaluate(model, num_epochs, train_loader, test_loader, optimizer, criterion, device="cpu", is_inception=False):
  max_val_acc = 0
  e_measures = []
  pbar = tqdm(range(1,num_epochs+1))
  for e in pbar:
      losses =  train_epoch(model, train_loader, optimizer, criterion, device, is_inception)
      measures_on_train = eval_model(model, train_loader, device, is_inception)
      measures_on_test  = eval_model(model, test_loader, device, is_inception )
      train_loss = np.mean(losses)
      measures = {'epoch': e, 'train_loss': train_loss, 'train_acc' : measures_on_train['acc'].round(4), 'val_acc' : measures_on_test['acc'].round(4) }
      if (max_val_acc < measures_on_test['acc'].round(4)):
        
        max_val_acc = measures_on_test['acc'].round(4)
        torch.save(model.state_dict(), 'modelo')

      pbar.set_postfix(measures)     
      e_measures += [measures]
  return pd.DataFrame(e_measures)   




import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

class FaceMaskData(pl.LightningDataModule):
    
    def __init__(self, batch_size=32, data_url = None, data_dir: str = 'data', input_size=None, train_size=0.8, val_size=0.1, test_size=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_url = data_url
        self.input_size = input_size
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        # Augmentation policy for training set
        self.augmentation = transforms.Compose([
              transforms.RandomResizedCrop(size=self.input_size),
              transforms.RandomRotation(degrees=15),
              transforms.RandomHorizontalFlip(),              
              transforms.ToTensor(),
              #transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose([
              transforms.Resize(size=self.input_size),
              transforms.CenterCrop(size=self.input_size),
              transforms.ToTensor(),
              #transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

    def prepare_data(self):
        pass

    def setup(self, stage="train"):
        # build dataset
        data = datasets.ImageFolder(self.data_dir)
        self.num_classes = len(data.classes)
        # split dataset
        train_size_ = int(len(data) * self.train_size)
        val_size_   = int(len(data) * self.val_size)
        test_size_  = len(data) - train_size_ - val_size_
        self.train, self.val, self.test = random_split(data, [train_size_, val_size_, test_size_])                
        self.train.dataset.transform = self.augmentation
        self.val.dataset.transform = self.transform
        self.test.dataset.transform = self.transform
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
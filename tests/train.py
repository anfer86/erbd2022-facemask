import pytorch_lightning as pl
pl.seed_everything(42, workers=True)

from FaceMaskModel import FaceMaskModel
model_name = "squeezenet"
num_classes = 2

faceMaskModel = FaceMaskModel(
    model_name=model_name,
    num_classes=num_classes
)

from FaceMaskData import FaceMaskData
faceMaskData = FaceMaskData(input_size=faceMaskModel.input_size)
faceMaskData.setup()

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
checkpoint_callback = ModelCheckpoint(dirpath = None, save_top_k=1, monitor="val_loss", mode="min")
checkpoint_callback.dirpath = None

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
logger_csv = CSVLogger(save_dir="logs/",  name=model_name)
logger = TensorBoardLogger(save_dir="logs/",  name=model_name)

trainer = pl.Trainer(logger=[logger, logger_csv], callbacks=[early_stop_callback,checkpoint_callback], max_epochs=3, default_root_dir="models/")
trainer.fit(model=faceMaskModel, datamodule=faceMaskData)
trainer.test(datamodule=faceMaskData)


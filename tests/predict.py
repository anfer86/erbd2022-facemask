from FaceMaskModel import FaceMaskModel

model_name = "squeezenet"
faceMaskModel = FaceMaskModel.load_from_checkpoint('logs/{}/version_0/checkpoints/epoch=2-step=39.ckpt'.format(model_name))
print(faceMaskModel)

from FaceMaskData import FaceMaskDataModule
faceMaskData = FaceMaskDataModule(input_size=faceMaskModel.input_size)
faceMaskData.setup()

X_test, y_test = next(iter( faceMaskData.test_dataloader() ))
y_pred = faceMaskModel(X_test)
print(y_pred)
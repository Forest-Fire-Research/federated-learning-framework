from pandas import DataFrame
from torch.utils.data import TensorDataset, DataLoader

from utils.dataset import Dataset
from utils.stas import STASDataGenerator
from utils.target_types import DTarget
from utils.models import STASGeneralModel

class FederatedNode():
    def __init__(
            self, 
            _id:int,
            d:DataFrame,
            f:DataFrame,
            d_type:Dataset,
            d_target:DTarget,
            n:int,
            m:int,
            k:int=1,
            learning_rate:float=0.001, 
            batch_size:int=2048,
            workers:int=4,
            multiplyer:int=1,
    ):
        self.id = _id
        print(f"Initializing id: {self.id:3}")

        # init stas sampler
        stas_sampler = STASDataGenerator(
            d=d,
            f=f,
            k=k,
            n=n,
            m=m,
            d_type=d_type,
            d_target=d_target,
        )

        # build train and test data
        train_data = TensorDataset(
            stas_sampler.train_x,
            stas_sampler.train_y
        )
        test_data = TensorDataset(
            stas_sampler.test_x,
            stas_sampler.test_y
        )
        del stas_sampler

        # build dataloaders
        self.train_dataloader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=True
        )
        self.test_dataloader = DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=False
        )
        del train_data
        del test_data

        # build model
        feature_count = (n+1) * len(d_type.value['data_columns'])
        self.model = STASGeneralModel(
            num_features=feature_count, 
            target_type=d_target,
            learning_rate=learning_rate,
            multiplyer=multiplyer,
        )

        # print(f"Initialization Finished for id: {self.id:3}")
    
    def set_lr(self, lr):
        self.model.set_lr(lr=lr)

    def get_model(self):
        return self.model

    def set_model(self, model:STASGeneralModel):
        self.model = model

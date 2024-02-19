from torch.optim import Adam
from torch.nn.functional import mse_loss
from torch.nn import Sigmoid, LeakyReLU, Linear, ModuleList
from pytorch_lightning import LightningModule
from torchmetrics import MeanSquaredError
from torchmetrics.classification import BinaryF1Score

from utils.target_types import DTarget

class STASGeneralModel(LightningModule):
    def __init__(
            self, 
            num_features:int, 
            target_type:DTarget,
            relu_slope:float=0.01, 
            learning_rate:float=0.01
    ):
        super().__init__()
        # parameter intilization
        self.relu_slope = relu_slope
        self.target_type = target_type
        self.learning_rate = learning_rate

        # init metrics
        if self.target_type == DTarget.BOOLEAN:
            self.f1_score = BinaryF1Score()
        elif self.target_type == DTarget.AREA:
            self.rmse = MeanSquaredError(squared=False)

        # build layers
        self.model = ModuleList()
        self.__add_linear_hidden_block(
            in_features=int(1.00*num_features),
            out_features=int(1.00*num_features)
        ),
        self.__add_linear_hidden_block(
            in_features=int(1.00*num_features),
            out_features=int(0.75*num_features)
        ),
        self.__add_linear_hidden_block(
            in_features=int(0.75*num_features),
            out_features=int(0.50*num_features)
        ),
        self.__add_linear_hidden_block(
            in_features=int(0.50*num_features),
            out_features=int(0.25*num_features)
        ),
        self.__add_output_block(
            in_features=int(0.25*num_features)
        )
    
    def __add_linear_hidden_block(self, in_features, out_features):
        self.model.append(
            Linear(
                in_features=in_features,
                out_features=out_features,
            )
        )
        self.model.append(
            LeakyReLU(negative_slope=self.relu_slope)
        )
        return 
    
    def __add_output_block(self, in_features):
        self.model.append(
            Linear(
                in_features=in_features,
                out_features=1,
            )
        )
        if self.target_type == DTarget.BOOLEAN:
            self.model.append(Sigmoid())
        elif self.target_type == DTarget.AREA:
            self.model.append(LeakyReLU(negative_slope=self.relu_slope))

        return 

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
            
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = mse_loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = mse_loss(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        if self.target_type == DTarget.BOOLEAN:
            self.f1_score(y_hat, y)
            self.log('f1_score', self.f1_score, on_epoch=True)
        elif self.target_type == DTarget.AREA:
            self.rmse(y_hat, y)
            self.log('rmse', self.rmse, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

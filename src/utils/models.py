from torch import nan_to_num
from torch.optim import Adam, SGD
from torch.nn.functional import mse_loss, binary_cross_entropy
from torch.nn import Sigmoid, LeakyReLU, Linear, ModuleList, BatchNorm1d, Tanh
from pytorch_lightning import LightningModule
from torchmetrics import MeanSquaredError, R2Score
from torchmetrics.classification import BinaryF1Score

from utils.target_types import DTarget



class STASGeneralModel(LightningModule):
    def __init__(
            self, 
            num_features:int, 
            target_type:DTarget,
            relu_slope:float=0.01, 
            learning_rate:float=0.01,
            multiplyer:int=1
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
            self.r2_score = R2Score()

        # build layers
        self.model = ModuleList()
        self.__add_linear_hidden_block(
            in_features=int(1.00*num_features),
            out_features=int(1.00*num_features*multiplyer)
        ),
        self.__add_linear_hidden_block(
            in_features=int(1.00*num_features*multiplyer),
            out_features=int(0.75*num_features*multiplyer)
        ),
        self.__add_linear_hidden_block(
            in_features=int(0.75*num_features*multiplyer),
            out_features=int(0.50*num_features*multiplyer)
        ),
        self.__add_linear_hidden_block(
            in_features=int(0.50*num_features*multiplyer),
            out_features=int(0.25*num_features*multiplyer)
        ),
        self.__add_output_block(
            in_features=int(0.25*num_features*multiplyer)
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
        self.model.append(
            BatchNorm1d(out_features)
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
        # elif self.target_type == DTarget.AREA:
        #     self.model.append(Tanh())
            # self.model.append(LeakyReLU(negative_slope=self.relu_slope))

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
        if self.target_type == DTarget.BOOLEAN:
            loss = binary_cross_entropy(y_hat, y)
        elif self.target_type == DTarget.AREA:
            loss = mse_loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(nan_to_num(x))
        if self.target_type == DTarget.BOOLEAN:
            loss = binary_cross_entropy(y_hat, y)
        elif self.target_type == DTarget.AREA:
            loss = mse_loss(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        if self.target_type == DTarget.BOOLEAN:
            self.f1_score(y_hat, y)
            self.log('f1_score', self.f1_score, on_epoch=True)
        elif self.target_type == DTarget.AREA:
            self.rmse(y_hat, y)
            self.r2_score(y_hat, y)
            self.log('rmse', self.rmse, on_epoch=True)
            self.log('r2_score', self.r2_score, on_epoch=True)
        return loss

    def set_lr(self, lr:float) -> None:
        self.learning_rate = lr
        self.configure_optimizers()

    def configure_optimizers(self):
        if self.target_type == DTarget.BOOLEAN:
            optimizer = SGD(self.parameters(), lr=self.learning_rate)
        elif self.target_type == DTarget.AREA:
            optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

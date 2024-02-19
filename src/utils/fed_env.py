from sqlalchemy.engine import URL
from numpy.random import choice

from utils.generate_subdivision import GenSubdivision
from utils.models import STASGeneralModel
from utils.target_types import DTarget
from utils.fed_node import FederatedNode
from utils.dataset import Dataset

import mlflow
from copy import deepcopy
from torch import no_grad
from pytorch_lightning import Trainer

from torch import set_float32_matmul_precision
set_float32_matmul_precision('high')


class FederatedEnvironment():
    def __init__(
            self, 
            d_type:Dataset,
            d_target:DTarget,
            db_url:URL,
            mlflow_uri:str,
            n:int,
            m:int,
            k:int=1, 
            num_conc_node_train:int=3, 
            learning_rate:float=0.001,
            batch_size:float=2048,
    )->None:

        self.k = k
        self.n = n
        self.m = m
        self.nodes = []

        self.d_type = d_type
        self.d_target = d_target
        self.learning_rate = learning_rate
        self.num_conc_node_train = num_conc_node_train

        # load D 
        d_data_generator = GenSubdivision(
            d_full = self.d_type,
            db_url = db_url
        )
        d_map = d_data_generator.gen_subdivisions()
        del d_data_generator

        # load F
        fire_generator = GenSubdivision(
            d_full = Dataset.FIRE,
            db_url = db_url
        )
        fire_d_map = fire_generator.gen_subdivisions()
        del fire_generator

        # build nodes
        for (d_s_id, d), (f_s_id, f) in zip(d_map,fire_d_map):
            assert d_s_id == f_s_id, F"D and F are not zipped with same S id"
            node = FederatedNode(
                _id=d_s_id,
                d=d,
                f=f,
                d_type=self.d_type,
                d_target=d_target,
                n=n,
                m=m,
                k=k,
                learning_rate=self.learning_rate, 
                batch_size=batch_size,
            )
            self.nodes.append(node)

        # init mlflow
        self.mlflow_uri = mlflow_uri
        self.__init_mlflow__()

        # init global model 
        self.reset_global_model()
    
    def get_mlflow_exp_name(self):
        return f"COMPSAC_24_FL_{self.d_type.name}"

    def __init_mlflow__(self):        
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.get_mlflow_exp_name())

    def get_num_features(self):
        return (self.n+1) * len(self.d_type.value['data_columns'])

    def reset_global_model(self)->None:
        feature_count = self.get_num_features()
        self.global_model = STASGeneralModel(
            num_features=feature_count, 
            target_type=self.d_target,
            learning_rate=self.learning_rate
        )

    def get_global_test_loader(self):
        num_nodes = self.get_node_count()
        for node_index in range(num_nodes):
            loader = self.nodes[node_index].test_dataloader
            for batch in loader:
                yield batch

    def get_phase_3_metrics(self)->dict:
        print("Phase 3 testing")
        # create lighting trainer
        trainer = Trainer(
            max_epochs=1, 
            accelerator="gpu", 
            enable_progress_bar=True, 
            logger=False,
            callbacks=[]
        )

        # run validation loop on trainer
        metrics = trainer.validate(
            model=self.global_model, 
            dataloaders=self.get_global_test_loader()
        )
        return_metrics = {}
        for metric in metrics[0].keys():
            return_metrics[f"Phase_3_{metric}"] = metrics[0][metric]
        del trainer
        del metrics
        return return_metrics

    def get_phase_2_metrics(self, node_indexes:list):
        print("Phase 2 testing")
        metrics = {} 
        # create trainer
        trainer = Trainer(
            max_epochs=1, 
            accelerator="gpu", 
            check_val_every_n_epoch=1, 
            logger=False, 
            callbacks=[],
            enable_progress_bar=False
        )
        # get global train loader
        global_train_loader = self.get_global_test_loader()
        # get metrics for each node in list "node_indexes"
        for node_index in node_indexes:
            node = self.nodes[node_index]
            node_metrics = trainer.validate(
                model=node.get_model(), 
                dataloaders=global_train_loader
            )
            for metric in node_metrics[0].keys():
                metrics[f"Phase_2_{metric}_{node.id}"] = node_metrics[0][metric]
        
        del global_train_loader
        return metrics

    def get_phase_1_metrics(self, node_index:int):
        print(f"Phase 1 testing in node {node_index}")
        # get node
        node = self.nodes[node_index]
        # create lighting trainer
        trainer = Trainer(
            max_epochs=1, 
            accelerator="gpu", 
            enable_progress_bar=True, 
            logger=False,
            callbacks=[]
        )
        # run validation loop on trainer
        metrics = trainer.validate(
            model=node.get_model(), 
            dataloaders=node.test_dataloader
        )
        return_metrics = {}
        for metric in metrics[0].keys():
            return_metrics[f"Phase_1_{metric}_{node_index}"] = metrics[0][metric]
        del trainer
        del metrics
        return return_metrics

    def get_global_model(self)->STASGeneralModel:
        feature_count = self.get_num_features()
        # creating a default copy of the model 
        model = STASGeneralModel(
            num_features=feature_count, 
            target_type=self.d_target,
            learning_rate=self.learning_rate
        )
        # loading a DEEPCOPY of the state dict into the model
        model.load_state_dict(deepcopy(self.global_model.state_dict()))
        return model
    
    def get_nodes(self, node_indexes:list)->list:
        return [self.nodes[index] for index in node_indexes]

    def get_node_count(self)->int:
        return len(self.nodes)

    def randomly_sample_nodes(self, n:int)->list:
        num_nodes = self.get_node_count()
        node_indexes = list(range(num_nodes))
        selected_node_index = choice(
            node_indexes, 
            n, 
            replace=False
        )
        return list(selected_node_index)

    def __reset_models(self)->None:
        # reset global model
        self.reset_global_model()

        node_indexes = list(range(len(self.nodes)))
        # update clinets to global model
        self.update_node_models(node_indexes)


    def train_nodes(
            self, 
            node_indexes:list, 
            epochs:int=20, 
            reset:bool=False, 
            log:bool=False
    )->None:
        if reset:
            # reset all models
            self.__reset_models()
        
        # train all node models
        for node_index in node_indexes:
            # get node
            node = self.nodes[node_index]

            print(f"\n++++++++++++++++++++++++++   Training node {node.id:2.0f}   +++++++++++++++++++++++++++++++++++++++++++++++++++++")

            # Initialize a trainer
            trainer = Trainer(
                max_epochs=epochs, 
                accelerator="gpu", 
                check_val_every_n_epoch=1, 
                logger=False, 
                enable_progress_bar=True,
                callbacks=[]
                )

            if log:
                # Auto log all MLflow entities
                # mlflow.pytorch.autolog()
                with mlflow.start_run(nested=True):
                    # set model tages
                    mlflow.set_tags(
                        {
                            'is_node':True, 
                            'node_id': node.id, 
                            'target': self.d_type.name,
                            'd_type': self.d_type.name,
                            'train_size': len(node.train_dataloader.dataset),
                            'test_size': len(node.test_dataloader.dataset),
                        }
                    )
                    # fit model
                    trainer.fit(
                        model=node.get_model(), 
                        train_dataloaders=node.train_dataloader, 
                        val_dataloaders=node.test_dataloader
                    )
                    # end mlflow logging
                    mlflow.end_run()
            else:
                # only run training 
                trainer.fit(model=node.get_model(), train_dataloaders=node.train_dataloader)
            del trainer

    def update_node_models(self, indexes)->None:
        print("\nStarted updating nodes")
        for index in indexes:
            node = self.nodes[index]

            global_model = self.get_global_model()

            node.set_model(global_model)
            print(f"Updating node {node.id}")
        print("\n")

    def aggregrate_global_model(self, node_indexes:list):
        for layer_index, __layer in enumerate(self.global_model.model):
            try:
                # load layer 'weights' and 'bias'
                gloabl_layer_weight, gloabl_layer_bias = self.global_model.model[layer_index].parameters()

                with no_grad():
                    # average weights and bias for all node models
                    for node_index in node_indexes:
                        node_model = self.nodes[node_index].get_model()
                        node_layer_weight, node_layer_bias = node_model.model[layer_index].parameters()
                        gloabl_layer_weight += node_layer_weight
                        gloabl_layer_bias += node_layer_bias
                    gloabl_layer_weight /= (len(node_indexes) + 1)
                    gloabl_layer_bias /= (len(node_indexes) + 1) 
                
                # set gradient to True for future training
                gloabl_layer_weight.requires_grad = True
                gloabl_layer_bias.requires_grad = True
            except Exception as e:
                print(f"{__layer} --> {e}")
                continue

    def fed_avg(
            self, 
            epochs:int=1000, 
            num_nodes_per_epoch:int=2,
            num_training_per_epoch:int=1
    )->None:
        # reset global model 
        self.reset_global_model()
        
        # set up mlflow tracking
        with mlflow.start_run():
            # tag run
            mlflow.set_tags({
                'is_node':False, 
                'node_id': -1, 
                'target': self.d_target.name,
                'd_type': self.d_type.name,
                'train_size': sum([len(node.train_dataloader.dataset)for node in self.nodes]),
                'test_size': sum([len(node.test_dataloader.dataset)for node in self.nodes]),
                'num_nodes_per_epoch': num_nodes_per_epoch,
                'num_training_per_epoch':num_training_per_epoch,
            })
            
            # start fed average training
            for epoch in range(epochs):
                # select nodes 
                node_indexes = self.randomly_sample_nodes(n=num_nodes_per_epoch)

                # update all node models
                self.update_node_models(indexes=node_indexes)
                
                # train nodes 
                self.train_nodes(
                    node_indexes=node_indexes, 
                    epochs=num_training_per_epoch
                )

                # update global model weight using fed average
                self.aggregrate_global_model(node_indexes=node_indexes)
                
                # get global model metrics
                global_metrics = self.get_phase_3_metrics()
                mlflow.log_metrics(metrics=global_metrics, step=epoch)

                # get local model metrics on global data
                all_node_index = list(range(self.get_node_count()))
                node_metrics = self.get_phase_2_metrics(all_node_index)
                mlflow.log_metrics(metrics=node_metrics, step=epoch)

            # # log models
            # mlflow.pytorch.log_model(self.get_global_model(), "model")


            # end mlflow run
            mlflow.end_run()
        

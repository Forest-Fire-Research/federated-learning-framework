from pandas import DataFrame, DateOffset, merge
from torch import tensor, float32
import sys
from utils.dataset import Dataset
from utils.target_types import DTarget

class STASDataGenerator():
    def __init__(
            self,
            k:int,
            n:int,
            m:int,
            d_target:DTarget,
            d_station:bool,
            d:DataFrame,
            d_type:Dataset,
            f:DataFrame,
    ) -> None:
        self.k = k
        self.n = n
        self.m = m
        self.d_target = d_target
        self.d_station = d_station
        self.d_type = d_type
        self.d = d
        self.f = f

        self.generate()

    def get_processed_data(self):
        d_p = None
        if self.d_station:
            d_p = self._get_k_nearest_station_data()
        else:
            d_p = self._get_aggregrated_dataset()
        # adding extra compuation here to reduce repeating the same operation late 
        d_p = merge(
            left = self.f,
            right = self.d,
            on = 'division_id'
        )
        return d_p

    def _get_k_nearest_station_data(self):
        pass

    def _get_aggregrated_dataset(self):
        # to save on computation this step was completedd in the preprocessing notebooks
        assert self.d_type == Dataset.LIGHTNING, f"ONLY event-based data can be passed to this function!"
        return self.d
        
    def get_fire_points(self, d_p):
        e_f = d_p[(d_p["start_date"] <= d_p["timestamp"]) & (d_p["start_date"] >= d_p["timestamp"]-DateOffset(days=self.n))]
        e_f = e_f.groupby(Dataset.FIRE.value['data_columns'])
        return e_f
    
    def get_non_fire_points(self, d_p):
        e_nf = d_p[(d_p["start_date"] <= d_p["timestamp"]-DateOffset(months=self.m)) & (d_p["start_date"] >= d_p["timestamp"]-DateOffset(months=self.m,days=self.n))]
        e_nf = e_nf.groupby(Dataset.FIRE.value['data_columns'])
        return e_nf
    
    def __del_spatio_temporal_info(self, df:DataFrame):
        data_columns = self.d_type.value['data_columns']
        df = df[data_columns]
        del data_columns
        return df
    
    def __flatten_data(self, df:DataFrame):
        flattend_df = df.stack().reset_index(drop=True).to_list()
        return flattend_df

    def get_dataset(self, e_f, e_nf):
        # initializa dataset
        dataset = []
        targets = []

        # push e_f to datset
        for (_fire_date, area_burn), datapoint_df in e_f:
            # append target value 
            if self.d_target == DTarget.AREA:
                targets.append(area_burn)
            elif self.d_target == DTarget.BOOLEAN:
                targets.append(1)
            else:
                targets.append(None)
            # deleted spatial or teporal indo
            datapoint_df = self.__del_spatio_temporal_info(datapoint_df)
            datapoint = self.__flatten_data(datapoint_df)
            del datapoint_df
            # append data_points 
            dataset.append(datapoint)
            del datapoint

        # push e_nf to datset
        for (_fire_date, area_burn), datapoint_df in e_nf:
            # append target value 
            if (self.d_target == DTarget.AREA) or (self.d_target == DTarget.BOOLEAN):
                targets.append(0)
            else:
                targets.append(None)
            # deleted spatial or teporal indo
            datapoint_df = self.__del_spatio_temporal_info(datapoint_df)
            datapoint = self.__flatten_data(datapoint_df)
            del datapoint_df
            # append data_points 
            dataset.append(datapoint)
            del datapoint
        
        self.target_df = DataFrame({
            "target": targets
        })
        dataset_columns = [f"{column}_{n}" for n in range(self.n+1) for column in self.d_type.value['data_columns']]
        self.dataset_df = DataFrame(
            data = dataset,
            columns = dataset_columns
        )
        del dataset
        del targets
        del dataset_columns
        self.dataset_df.fillna(0, inplace=True)

        # randomly sample train and test data
        self.train_index = self.dataset_df.sample(frac=0.8).index
        self.test_index = self.dataset_df.drop(self.train_index).index
        
        # standardize the data 
        self.mean = self.dataset_df.loc[self.train_index].mean()
        self.std = self.dataset_df.loc[self.train_index].std()
        self.dataset_df = (self.dataset_df - self.mean) / self.std

        # split into datasets
        self.train_x = tensor(
            self.dataset_df.loc[self.train_index].values, 
            dtype=float32
        )
        self.train_y = tensor(
            self.target_df.loc[self.train_index].values, 
            dtype=float32
        )
        self.test_x = tensor(
            self.dataset_df.loc[self.test_index].values,
            dtype=float32
        )
        self.test_y = tensor(
            self.target_df.loc[self.test_index].values,
            dtype=float32
        )

        # discard excess info
        del self.dataset_df
        del self.target_df
        del self.test_index
        del self.train_index

    def generate(self):
        d_p = self.get_processed_data()
        e_f = self.get_fire_points(d_p = d_p)
        e_nf = self.get_non_fire_points(d_p = d_p)
        self.get_dataset(e_f=e_f, e_nf=e_nf)

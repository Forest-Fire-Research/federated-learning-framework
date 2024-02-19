from pandas import DataFrame, DateOffset, merge, date_range
from concurrent.futures import ThreadPoolExecutor
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
            d:DataFrame,
            d_type:Dataset,
            f:DataFrame,
    ) -> None:
        """ STAS Constructor

        Args:
            k (int): Num of nearest stations
            n (int): Num of timestams to consider in a datapoint
            m (int): Num of months to look back at to extract non-fire points
            d_target (DTarget): Target value type
            d (DataFrame): Dataset
            d_type (Dataset): Dataset type
            f (DataFrame): Fire dataset
        """
        self.k = k
        self.n = n
        self.m = m
        self.d_target = d_target
        self.d_type = d_type
        self.d = d
        self.f = f

        self.generate()

    def get_processed_data(self) -> DataFrame:
        """ Generate processed data

        Returns:
            DataFrame: d_p (preprocessed dataset)
        """
        d_p = None
        if self.d_type == Dataset.WEATHER:
            d_p = self._get_k_nearest_station_data()
        elif self.d_type == Dataset.LIGHTNING:
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

    def _get_aggregrated_dataset(self) -> DataFrame:
        """ Aggregrate the datapoints in the dataset

        Returns:
            DataFrame: Aggregrated dataset
        """
        # to save on computation this step was completedd in the preprocessing notebooks
        assert self.d_type == Dataset.LIGHTNING, f"ONLY event-based data can be passed to this function!"
        return self.d
    
    def __get_start_time_offset(
        self,
        is_fire:bool,
    ) -> DateOffset:
        """ Get start offset

        Args:
            is_fire (bool): Specifes if fire events or non-fire events

        Returns:
            DateOffset: Start offset
        """
        offset = self.__get_start_fire_offset() if is_fire else self.__get_start_non_fire_offset()
        return offset

    def __get_start_fire_offset(self) -> DateOffset:
        return DateOffset(days = self.n)

    def __get_start_non_fire_offset(self) -> DateOffset:
        return DateOffset(days = self.n, months = self.m)

    def __get_end_time_offset(
        self,
        is_fire:bool,
    ) -> DateOffset:
        """ Get end offset

        Args:
            is_fire (bool): Specifes if fire events or non-fire events

        Returns:
            DateOffset: End offset
        """
        offset = self.__get_end_fire_offset() if is_fire else self.__get_end_non_fire_offset()
        return offset

    def __get_end_fire_offset(self):
        return DateOffset(months = 0)

    def __get_end_non_fire_offset(self):
        return DateOffset(months = self.m)

    def _filter_to_timeseries(
        self, 
        df:DataFrame, 
        is_fire:bool
    ) -> DataFrame:
        """ Filter the cross join points to be in the timeseries range

        Args:
            df (DataFrame): Preporcessed crossjoin dataset
            is_fire (bool): Specifes if the filter is for fire events or non-fire events

        Returns:
            DataFrame: Filtered preprocessed event datapoints
        """
        d_date_column = self.d_type.value['date_column']
        f_date_column = Dataset.FIRE.value['date_column']
        start_offset = self.__get_start_time_offset(is_fire=is_fire)
        end_offset = self.__get_end_time_offset(is_fire=is_fire)
        df = df[
            (df[f_date_column] - start_offset <= df[d_date_column]) &
            (df[d_date_column] <= df[f_date_column] - end_offset)
        ]
        del d_date_column
        del f_date_column
        del end_offset
        del start_offset
        return df

    def _fiter_group_wrapper(
        self, 
        df:DataFrame, 
        is_fire:bool
    ):
        """ Wrapper to group fitered event datapoints

        Args:
            df (DataFrame): Preporcessed crossjoin dataset
            is_fire (bool): Specifes if the filter is for fire events or non-fire events

        Returns:
            DataFrameGroupBy: Key value map of datapoints
        """
        f_data_columns = Dataset.FIRE.value['data_columns']
        df = self._filter_to_timeseries(
            df = df,
            is_fire = is_fire
        )
        df = df.groupby(f_data_columns)
        del f_data_columns
        return df
        
    def get_fire_points(
        self, 
        d_p:DataFrame
    ):
        """ Returns fire event point map

        Args:
            d_p (DataFrame): Preprocessed dataset

        Returns:
            DataFrameGroupBy: Key value map of fire event datapoints
        """
        e_f = self._fiter_group_wrapper(
            df = d_p,
            is_fire = True
        )
        return e_f
    
    def get_non_fire_points(
        self, 
        d_p:DataFrame
    ):
        """ Returns non-fire event point map

        Args:
            d_p (DataFrame): Preprocessed dataset

        Returns:
            DataFrameGroupBy: Key value map of non-fire event datapoints
        """
        e_nf = self._fiter_group_wrapper(
            df = d_p,
            is_fire = False
        )
        return e_nf
    
    def __order_by_time(
        self,
        df:DataFrame,
        is_fire:bool,
        start_date
    ) -> DataFrame:
        """ Order event datapoint's data by datatime

        Args:
            df (DataFrame): Event datapoint df

        Returns:
            DataFrame: Ordered event datapoint event
        """
        d_date_column = self.d_type.value['date_column']
        start_offset = self.__get_start_time_offset(is_fire=is_fire)
        end_offset = self.__get_end_time_offset(is_fire=is_fire)

        date_df = DataFrame({
            d_date_column: date_range(
                start = start_date - start_offset, 
                end = start_date - end_offset, 
                freq='D'
            )
        })
        df = merge(
            date_df,
            df,
            on = d_date_column,
            how = 'left'
        ).fillna(0)

        del d_date_column
        del start_offset
        del end_offset
        del date_df

        return df

    def __del_spatio_temporal_info(
        self, 
        df:DataFrame
    ) -> DataFrame:
        """ Delete any spatial or temporal information

        Args:
            df (DataFrame): Processed data

        Returns:
            DataFrame: Data without spatial or temporal information
        """
        data_columns = self.d_type.value['data_columns']
        df = df[data_columns]
        del data_columns
        return df
    
    def __flatten_data(
        self, 
        df:DataFrame
    ) -> list:
        """ Flatten the datapoint df to a list 

        Args:
            df (DataFrame): Datapoint df

        Returns:
            list: Flattned datapoint
        """
        df = df.stack().reset_index(drop=True).to_list()
        return df

    def __process_fire_events(
        self,
        param
    ) -> list:
        """ Datapoint processer

        Args:
            param (DataFrameGroupBy): Event datapoint key and value 

        Returns:
            list: Processed event datapoint
        """
        return self.__process_datapoints(
            param = param,
            is_fire = True 
        )
    
    def __process_non_fire_events(
        self,
        param
    ) -> list:
        """ Datapoint processer

        Args:
            param (DataFrameGroupBy): Event datapoint key and value 

        Returns:
            list: Processed event datapoint
        """
        return self.__process_datapoints(
            param = param,
            is_fire = False 
        )

    def __process_datapoints(
        self,
        param,
        is_fire
    ) -> list:
        """ Datapoint processer

        Args:
            param (DataFrameGroupBy): Event datapoint key and value 

        Returns:
            list: Processed event datapoint
        """
        (fire_date, _area_burnt), datapoint_df = param
        del param
        del _area_burnt

        # order data evnt date
        datapoint_df = self.__order_by_time(
            df = datapoint_df,
            is_fire = is_fire,
            start_date = fire_date
        )[:self.n + 1]
        del fire_date

        num_past_points = len(datapoint_df)
        assert num_past_points == self.n + 1, f"Only {num_past_points} points found instead of {self.n + 1}"
        
        # deleted spatial or teporal info 
        datapoint_df = self.__del_spatio_temporal_info(datapoint_df)
        datapoint = self.__flatten_data(datapoint_df)
        del datapoint_df

        return datapoint
        
    def get_dataset(
        self, 
        e_f, 
        e_nf
    ) -> None:
        """ Generate the dataset and store it internally

        Args:
            e_f (DataFrameGroupBy): Fire event datapoints 
            e_nf (DataFrameGroupBy): Non-fire event datapoints
        """
        # initializa dataset
        dataset = []
        targets = []

        # push e_f targets
        for (_fire_date, area_burn), datapoint_df in e_f:
            # append target value 
            if self.d_target == DTarget.AREA:
                targets[data_index] = area_burn
            elif self.d_target == DTarget.BOOLEAN:
                targets.append(1)
            else:
                targets.append(None)

        # push e_nf targets
        for (_fire_date, area_burn), datapoint_df in e_nf:
            # append target value 
            if (self.d_target == DTarget.AREA) or (self.d_target == DTarget.BOOLEAN):
                targets.append(0)
            else:
                targets.append(None)

        # push fire datapoints
        with ThreadPoolExecutor(max_workers = 8) as fire_executor:
            dataset += list(fire_executor.map(self.__process_fire_events, e_f))

        # push non-fire datapoints
        with ThreadPoolExecutor(max_workers = 8) as fire_executor:
            dataset += list(fire_executor.map(self.__process_non_fire_events, e_nf))
        
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
        # self.dataset_df.fillna(0, inplace=True)

        # randomly sample train and test data
        self.train_index = self.dataset_df.sample(frac=0.8).index
        self.test_index = self.dataset_df.drop(self.train_index).index
        
        # standardize the data 
        self.mean = self.dataset_df.loc[self.train_index].mean()
        self.std = self.dataset_df.loc[self.train_index].std()
        self.dataset_df = (self.dataset_df - self.mean) / self.std
        self.dataset_df.fillna(0, inplace=True)

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

    def generate(self) -> None:
        d_p = self.get_processed_data()
        e_f = self.get_fire_points(d_p = d_p)
        e_nf = self.get_non_fire_points(d_p = d_p)
        self.get_dataset(e_f=e_f, e_nf=e_nf)

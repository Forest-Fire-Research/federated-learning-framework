from sqlalchemy.engine import URL
from sqlalchemy import create_engine
from pandas import read_sql, DataFrame
from geopandas import read_postgis, GeoDataFrame
from utils.dataset import Dataset


class GenSubdivision():
    def __init__(
            self,
            d_full:Dataset,
            s:Dataset = Dataset.SUBDIVISION,
            db_url:URL = None
    ) -> None:
        self.engine = create_engine(db_url)
        self.d_full = d_full

    def __get_subdivion_data_query(self) -> str:
        query =  """SELECT * FROM "S";"""
        return query
    
    def __get_lightning_data_query(self) -> str:
        query =  """SELECT * FROM "L_s";"""
        return query 
    
    def __get_fire_data_query(self) -> str:
        query = """ 
            SELECT
                fs.division_id,
                fs.start_date,
                fs.area_burnt_ha
            FROM 
                "F_s" fs
            WHERE
                fs.cause = 'L'
        """
        return query
    
    def __get_weather_data_query(self) -> str:
        query = ""
        return query
    
    def __read_geodata(
            self, 
            query:str, 
            index_col:str = None, 
            geom_col:str = 'geometry',
            crs:str = "EPSG:4326"
    ) -> GeoDataFrame:
        data = read_postgis(
            sql = query,
            con = self.engine,
            geom_col = geom_col,
            index_col = index_col,
            crs = crs
        )
        return data
    
    def __read_data(
            self,
            query:str,
            index_col:str = None,
    ) -> DataFrame:
        data = read_sql(
            sql = query,
            con = self.engine,
            index_col = index_col
        )
        return data

    def get_subdivision_dataset(self) -> GeoDataFrame:
        subdivision_data_query = self.__get_subdivion_data_query()
        subdivion_data = self.__read_geodata(
            subdivision_data_query,
            index_col = Dataset.SUBDIVISION.value['index_col']
        )
        return subdivion_data
    
    def __get_data_query(self) -> str:
        """ Generates the query to get the appropriate data

        Raises:
            ValueError: The dataset type is invalid.

        Returns:
            str: dataset query
        """
        if self.d_full == Dataset.LIGHTNING:
            return self.__get_lightning_data_query()
        elif self.d_full == Dataset.WEATHER:
            return self.__get_weather_data_query()
        elif self.d_full == Dataset.FIRE:
            return self.__get_fire_data_query()
        else:
            raise ValueError("Invalid return dataset type!!!")
    
    def gen_subdivisions(self):
        data_query = self.__get_data_query()
        if self.d_full.value['is_geo']:
            data = self.__read_geodata(
                data_query,
                index_col = self.d_full.value['index_col']
            )
        else:
            data = self.__read_data(
                data_query,
                index_col = self.d_full.value['index_col']
            )
        d_map = data.groupby(
            by = self.d_full.value['index_col']
        )

        return d_map

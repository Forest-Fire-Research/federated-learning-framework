from sqlalchemy.engine import URL
from sqlalchemy import create_engine, inspect, text
from pandas import read_sql, DataFrame
from geopandas import read_postgis, GeoDataFrame
from utils.dataset import Dataset


class GenSubdivision():
    def __init__(
            self,
            n:int,
            m:int,
            k:int,
            d_full:Dataset,
            s:Dataset = Dataset.SUBDIVISION,
            db_url:URL = None
    ) -> None:
        self.n = n
        self.m = m
        self.k = k
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

    def __create_cache_table(
        self, 
        table_name:str,
        query:str,
    ) -> None:
        create_query = f"""
        CREATE TABLE IF NOT EXISTS "{table_name}"
        as (
            {query}
        );
        """

        with self.engine.connect() as con:
            con.execute(text(create_query))
            con.execute(text(f'ALTER TABLE "{table_name}" ADD PRIMARY KEY ("division_id", "climate_ID", "area_burnt_ha", "start_date");'))
            con.close()

    def __get_cache_k_nearest_table_query(self) -> str:
        query = f"""
        select 
            f.division_id,
            f.start_date,
            f.area_burnt_ha,
            w."climate_ID"
        from (
            select 
                division_id,
                start_date,
                area_burnt_ha,
                cause,
                geometry 
            from "F_s" as fs 
        ) as f
        cross join lateral (
            select 
                wm."climate_ID", 
                wm."geometry" <-> f."geometry" as distance
            from (
                select 
                    wms."climate_ID", 
                    wms."geometry",
                    wms."first_yr",
                    wms."last_yr"
                from "W_ms" as wms
            ) as wm
            where 
                EXTRACT(year FROM f.start_date) > wm.first_yr and 
                EXTRACT(year FROM f.start_date) < wm.last_yr  
            order by distance
            limit {self.k}
        ) as w
        """
        return query
    
    def __get_cache_weather_table_name(self) -> str:
        return f"W_m_k{self.k}"
    
    def __get_weather_data_query(self) -> str:
        table_name = self.__get_cache_weather_table_name()
        table_exists = inspect(self.engine).has_table(table_name, schema="public")
        if not table_exists:
            cache_query = self.__get_cache_k_nearest_table_query()
            self.__create_cache_table(
                table_name = table_name,
                query = cache_query
            )
            print(f"Table does not exist for weather cache k={self.k}!!! Creating table.")
        print(f"Cache Weather table found for k={self.k}!")
        query = f"""
            select 
                wmk.division_id,
                wmk.start_date,
                wmk.area_burnt_ha,
                wsk.date,
                wsk.extraterrestrial_irradiance,
                wsk.global_horizontal_irradiance,
                wsk.direct_normal_irradience,
                wsk.diffuse_horizontal_irradiance,
                wsk.global_horizontal_illumination_klux,
                wsk.direct_normal_illumination_klux,
                wsk.diffuse_horizontal_illumination_klux,
                wsk.zenith_illumination,
                wsk.sunlight_min,
                wsk.ceiling_height_meters,
                wsk.sky_layer_1,
                wsk.sky_layer_2,
                wsk.sky_layer_3,
                wsk.sky_layer_4,
                wsk.visibility_km,
                wsk.weather_thunderstorm,
                wsk.weather_rain,
                wsk.weather_drizzle,
                wsk.weather_snow_1,
                wsk.weather_snow_2,
                wsk.weather_ice,
                wsk.weather_visibility_1,
                wsk.weather_visibility_2,
                wsk.pressure_kpa,
                wsk.dry_bulb_temp_c,
                wsk.dew_point_temp_c,
                wsk.wind_direction_deg,
                wsk.wind_speed_mps,
                wsk.sky_cover,
                wsk.sky_cover_opaque,
                wsk.snow
            from "W_m_k1" wmk
            inner join "W_sp" wsk 
                on wsk.climate_id = wmk."climate_ID" 
            where 
                DATE(wsk.date) <= DATE(wmk.start_date) and 
                DATE(wmk.start_date) - make_interval(days => 7) - make_interval(months => 1)<= DATE(wsk.date)
        """
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

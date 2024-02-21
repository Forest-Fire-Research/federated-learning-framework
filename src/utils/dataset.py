from enum import Enum

# define an dataset enum
class Dataset(Enum):
    FIRE = {
        'id': 'fire',
        'index_col': 'division_id',
        'is_geo': False,
        'date_column': 'start_date',
        'data_columns': [
            'start_date', 
            'area_burnt_ha'
        ]
    }
    SUBDIVISION = {
        'id': 'subdivision',
        'index_col': "cid",
        'is_geo': True,
        'date_column': None,
        'data_columns': None
    }
    WEATHER = {
        'id': 'weather',
        'index_col': 'division_id',
        'is_geo': False,
        'date_column': 'date',
        'data_columns': [
            'extraterrestrial_irradiance',
            'global_horizontal_irradiance',
            'direct_normal_irradience',
            'diffuse_horizontal_irradiance',
            'global_horizontal_illumination_klux',
            'direct_normal_illumination_klux',
            'diffuse_horizontal_illumination_klux',
            'zenith_illumination',
            'sunlight_min',
            'ceiling_height_meters',
            'sky_layer_1',
            'sky_layer_2',
            'sky_layer_3',
            'sky_layer_4',
            'visibility_km',
            'weather_thunderstorm',
            'weather_rain',
            'weather_drizzle',
            'weather_snow_1',
            'weather_snow_2',
            'weather_ice',
            'weather_visibility_1',
            'weather_visibility_2',
            'pressure_kpa',
            'dry_bulb_temp_c',
            'dew_point_temp_c',
            'wind_direction_deg',
            'wind_speed_mps',
            'sky_cover',
            'sky_cover_opaque',
            'snow',
        ]
    }
    LIGHTNING = {
        'id': 'lightning',
        'index_col': 'division_id',
        'is_geo': False,
        'date_column': 'timestamp',
        'data_columns': [
            'multiplicity_sum', 
            'multiplicity_min', 
            'multiplicity_max', 
            'multiplicity_mean', 
            'event_strength_kiloamperes_mean', 
            'event_strength_kiloamperes_min',
            'event_strength_kiloamperes_max'
        ]
    }

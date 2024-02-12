from enum import Enum

# define an dataset enum
class Dataset(Enum):
    FIRE = {
        'id': 'fire',
        'index_col': "division_id",
        'is_geo': False,
        'data_columns': ["start_date", "area_burnt_ha"]
    }
    SUBDIVISION = {
        'id': 'subdivision',
        'index_col': "cid",
        'is_geo': True,
        'data_columns': []
    }
    WEATHER = {
        'id': 'weather',
        'index_col': "",
        'is_geo': False,
        'data_columns': []
    }
    LIGHTNING = {
        'id': 'lightning',
        'index_col': "division_id",
        'is_geo': False,
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

from enum import Enum

# define an dataset enum
class Dataset(Enum):
    FIRE = {
        'id': 'fire',
        'index_col': "division_id",
        'is_geo': False
    }
    SUBDIVISION = {
        'id': 'subdivision',
        'index_col': "cid",
        'is_geo': True
    }
    WEATHER = {
        'id': 'weather',
        'index_col': "",
        'is_geo': False
    }
    LIGHTNING = {
        'id': 'lightning',
        'index_col': "division_id",
        'is_geo': False
    }

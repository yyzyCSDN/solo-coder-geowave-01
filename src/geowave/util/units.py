from __future__ import annotations
_DISTANCE = {'m': 1.0, 'km': 1000.0, 'ft': 0.3048}
_TIME = {'s': 1.0, 'ms': 0.001, 'us': 1e-06}
_VELOCITY = {'m/s': 1.0, 'km/s': 1000.0, 'ft/s': 0.3048}

def convert(value: float, source: str, target: str) -> float:
    for table in (_DISTANCE, _TIME, _VELOCITY):
        if source in table and target in table:
            return value * table[source] / table[target]
    raise ValueError(f'incompatible or unknown units {source}->{target}')

def normalize_distance(value: float, unit: str) -> float:
    return convert(value, unit, 'm')

def normalize_time(value: float, unit: str) -> float:
    return convert(value, unit, 's')

def normalize_velocity(value: float, unit: str) -> float:
    return convert(value, unit, 'm/s')

def require_same_dimension(units: list[str]) -> str:
    if not units:
        raise ValueError('no units')
    tables = [name for name, table in (('distance', _DISTANCE), ('time', _TIME), ('velocity', _VELOCITY)) if all((u in table for u in units))]
    if len(tables) != 1:
        raise ValueError('mixed dimensions')
    return tables[0]

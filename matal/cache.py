import os
import pickle
import shutil
import hashlib
import base64
import json

import numpy as np
import pandas as pd

from .log import auto_log
from .configs import CACHE_DIR


class CacheLoadError(Exception):
    pass


class RobustJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, np.generic):
            return obj.tolist() 
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif hasattr(obj, 'serialize'):
            return f'*** unserializable object {repr(obj)} ***'
            # TODO: serialization for tensorflow.python.keras.engine.node.Node object
            # return obj.serialize()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return f'*** unserializable object {repr(obj)} ***'


def save_cache(obj: object, tag: str, version: str = None, meta: object = None, log=True) -> None:
    if version:
        cache_dir = os.path.join(CACHE_DIR, tag, version[:2])
    else:
        cache_dir = os.path.join(CACHE_DIR, tag)
    os.makedirs(cache_dir, exist_ok=True)
    
    if meta:
        json_name = f'cache_{version}.meta.json' if version else 'cache.meta.json'
        with open(os.path.join(cache_dir, json_name), 'w') as f:
            json.dump(meta, f, cls=RobustJSONEncoder, indent=2)

    cache_name = f'cache_{version}.pkl' if version else 'cache.pkl'
    with open(os.path.join(cache_dir, cache_name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
    if log: auto_log(f'Cache saved to {cache_name} @ {tag}', level='debug')

        
def load_cache(tag: str, version: str = None, log=True) -> object:
    if version:
        cache_dir = os.path.join(CACHE_DIR, tag, version[:2])
    else:
        cache_dir = os.path.join(CACHE_DIR, tag)
    cache_name = f'cache_{version}.pkl' if version else 'cache.pkl'
    
    try:
        with open(os.path.join(cache_dir, cache_name), 'rb') as f:
            obj = pickle.load(f)
            if log: auto_log(f'Cache loaded from {cache_name} @ {tag}', level='debug')
    except Exception:
        raise CacheLoadError()
    
    return obj


def obj_to_version(obj: object, size:int = 5) -> str:
    obj_pkl = pickle.dumps(obj)
    h = hashlib.blake2b(digest_size=size)
    h.update(obj_pkl)
    version = base64.b32encode(h.digest()).decode('utf-8').replace('=', '0')
    return version


def cache_return(func, save_meta=False, log=False):
    def wrapper(*args, **kwargs):
        tag = f'func__{func.__name__}'
        
        params = {'arg': args, 'kwargs': kwargs}
        version = obj_to_version(params, size=40)
        try:
            if log: auto_log(f'Try to load function cache for {tag} @ {version}', level='debug')
            obj = load_cache(tag, version=version)
            if log: auto_log(f'Loaded function cache from {tag} @ {version}')
        except CacheLoadError:
            obj = func(*args, **kwargs)
            save_cache(obj, tag, version=version, meta=params if save_meta else None)
            if log: auto_log(f'Saved function cache to {tag} @ {version}')
            
        return obj
    
    return wrapper
    

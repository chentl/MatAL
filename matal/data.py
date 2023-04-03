import base64
import hashlib
import json
import os
import pickle
import dill
import shutil
from typing import Union, List, Callable

import lz4
import lzma
import numpy as np
import pandas as pd

from .log import auto_log
from .configs import DATA_DIR, MODEL_DIR

DATASET_DIR = DATA_DIR / 'dataset' 


def tag_to_str(tags: Union[List[str], str]):
    if isinstance(tags, str):
        return tags
    elif isinstance(tags, list):
        return '__'.join(tags)


def load_dataset(name: str, tags: Union[List[str], str], fmt='csv',  **kwargs):
    tag_str = tag_to_str(tags)
    file_path = DATASET_DIR / f'{name}.{tag_str}.{fmt}'
    
    if fmt.startswith('csv'):
        return pd.read_csv(file_path, **kwargs)
    elif fmt.startswith('pkl'):
        return pd.read_pickle(file_path, **kwargs)
    else:
        raise ValueError()


def save_dataset(df: pd.DataFrame, name: str, tags: Union[List[str], str], fmt='csv', **kwargs):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    tag_str = tag_to_str(tags)
    file_path = DATASET_DIR / f'{name}.{tag_str}.{fmt}'
    
    if fmt.startswith('csv'):
        df.to_csv(file_path, **kwargs)
    elif fmt.startswith('pkl'):
        df.to_pickle(file_path, **kwargs)
    else:
        raise ValueError()


def load_data(name: str, reader: Union[Callable, str] = 'auto', mode: str = None, path: str = DATA_DIR, **kwargs):
    filepath = path / name
    
    if not callable(reader):
        assert reader == 'auto'
        ext = os.path.splitext(name)[-1]
        
        if ext == '.csv':
            func = pd.read_csv
            mode = None
        elif ext in ['.pk', '.pkl', '.pickle']:
            func = pickle.load
            mode = 'rb'
        elif ext in ['.dill', ]:
            func = dill.load
            mode = 'rb'
        elif ext == '.json':
            func = json.load
            mode = 'r'
        else:
            raise ValueError(f'Not supported ext: {ext}')
    else:
        func = reader
    
    if mode is None:
        return func(filepath, **kwargs)
    else:
        assert mode in ['r', 'rb']
        with open(filepath, mode) as f:
            return func(f, **kwargs)


def load_model(model, path=MODEL_DIR):
    for ext in ['pkl', 'dill']:
        if os.path.isfile(path / f'{model}.{ext}'):
            with open(path / f'{model}.{ext}', 'rb') as f:
                return dill.load(f)
        elif os.path.isfile(path / f'{model}.{ext}.lz4'):
            with lz4.frame.open(path / f'{model}.{ext}.lz4', 'rb') as f:
                return dill.load(f)
        elif os.path.isfile(path / f'{model}.{ext}.xz'):
            with lzma.open(path / f'{model}.{ext}.xz', 'rb') as f:
                return dill.load(f)
        else:
            continue
    raise FileNotFoundError(f'Model file {model} not found under {path}.')

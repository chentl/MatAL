import copy
import datetime
import glob
import json
import lzma
import os
import pickle
import platform
import random
import string
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import dill
import numpy as np
import pandas as pd
import scikeras
import sklearn
import skopt
import tensorflow as tf
from joblib import Parallel, delayed, parallel_backend
from scikeras.wrappers import KerasRegressor
from sklearn import metrics
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from sklearn.model_selection import cross_validate, KFold
from skopt.space import Real, Integer, Categorical

os.environ['MATAL_PROJ'] = 'AG'
PROJ_CODE = 'AG'
X_COLS = ['MMT', 'CNF', 'GEL', 'GLY']
X_TAGS = ['MMT', 'CNF', 'Gelatin', 'Glycerol']
X_UNITS = ['1', '1', '1', '1', ]

Y_COLS = ['VLTransmittance', 'UVTransmittance', 'IRTransmittance',
          'AshAreaRatio', 
          'Stress', 'Strain', 'Modulus', 'Toughness',
          'CurveAlpha', 'CurveBeta']
Y_TAGS = ['550nm transmittance', '365nm transmittance', '950nm transmittance',
          'Ash area ratio',
          'Strength', 'Strain', 'Young\'s modulus', 'Toughness', 
          'Curve alpha', 'Curve beta']
Y_UNITS = ['%', '%', '%',
           '1', 
           'MPa', '1', 'MPa', 'MPa', 
           '1', '1']
Y_NONLIN_LAST = ['sigmoid', 'sigmoid', 'sigmoid', 
                 'sigmoid', 
                 'relu', 'relu', 'relu', 'relu',
                 'sigmoid', 'sigmoid']
Y_SCALES = [100, 100, 100, 
            1, 
            150, 0.3, 10000, 6, 
            1, 1]

FINISHED_Y_COLS = ['Toughness']
# FINISHED_Y_COLS = ['VLTransmittance', 'UVTransmittance', 'IRTransmittance', 'AshAreaRatio', 'Toughness', 'CurveAlpha', 'CurveBeta']

AVAILABLE_DA_TAGS = ['da1000xy', 'da1000xys']

ROUND_START = 14
ROUND_END   = 15
MAX_N_JOBS = 9
MAX_N_MODELS_PER_JOB = 2
MAX_N_MODELS_PER_Y = 200
N_INIT_POINTS = 50
CV_N_FOLD = 5


from matal.utils import obj_to_version as obj_to_hash
from matal.utils import RobustJSONEncoder as NpEncoder
from matal.utils import auto_log
from matal.ml import build_smlp_model, plot_mlp_model


PROJ_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent / 'data'
RESULT_DIR = Path(__file__).parent / 'results' / PROJ_CODE


def get_sorted_job_list(n_calls=MAX_N_MODELS_PER_Y, max_calls=MAX_N_MODELS_PER_Y, max_jobs=MAX_N_JOBS):
    result_counts = []
    for r in range(ROUND_START, ROUND_END):
        rn = f'{r:02d}'
        for y in Y_COLS:
            if y in FINISHED_Y_COLS: continue
            prefix = f'{rn}_{y}'
            result_jsons = glob.glob(str(RESULT_DIR / prefix / f'{prefix}_*.result.json'))
            if len(result_jsons) < max_calls:
                result_counts.append([rn, y, len(result_jsons)])

    result_counts = sorted(result_counts, key=lambda l: l[2])
    jobs = [r[:2] + [i, min(n_calls + r[2], max_calls)] for i, r in enumerate(result_counts)]
    return jobs[:max_jobs]


def build_model(params):
    if params['early_stopping']:
        es_params = dict(callbacks={'es': tf.keras.callbacks.EarlyStopping},
                         callbacks__es__monitor='loss', 
                         callbacks__es__patience=100)
    else:
        es_params = dict()
    base_ann = KerasRegressor(model=build_smlp_model,
                              verbose=0, **es_params, **params)
    boost_type = params.get('boost', 'none')
    if boost_type == 'none':
        return base_ann
    elif boost_type.startswith('ada.'):
        n_est = int(boost_type.split('.')[1])
        return AdaBoostRegressor(base_estimator=base_ann, random_state=params['random_state'], n_estimators=n_est)
    elif boost_type.startswith('bag.'):
        n_est = int(boost_type.split('.')[1])
        return BaggingRegressor(base_estimator=base_ann, random_state=params['random_state'], n_estimators=n_est)
    else:
        raise ValueError()


def load_data(round_name, return_df=False, da_tag='da0', y=None):
    data_dir = DATA_DIR / f'{PROJ_CODE}'

    if isinstance(round_name, str):
        if round_name.isdigit():
            with open(
                    data_dir / f'round_{round_name}' / 'train_data' / f'{round_name}.train_data__{da_tag}.table.csv',
                    'r') as f:
                data = pd.read_csv(f)
        else:
            with open(data_dir / f'round_{round_name}' / 'raw_data' / f'{round_name}.raw_data.table.csv', 'r') as f:
                data = pd.read_csv(f)
    elif isinstance(round_name, int):
        with open(
                data_dir / f'round_{round_name:02d}' / 'train_data' / f'{round_name:02d}.train_data__{da_tag}.table.csv',
                'r') as f:
            data = pd.read_csv(f)
    else:
        raise ValueError(f'Unknown round_name: {round_name}')

    if return_df:
        return data
    else:
        _xs = [c for c in X_COLS if c in data.columns]
        _ys = [c for c in Y_COLS if c in data.columns]
        return data[_xs], data[_ys]


def random_str(l=8):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(l))


def random_int():
    r = random.Random()
    r.seed()
    return r.randint(0, 2 ** 32 - 1)


def list_to_str(l):
    s = []
    for li in l:
        if isinstance(li, float):
            s.append(f'{li:.1e}')
        else:
            s.append(str(li))
    return '[' + ', '.join(s) + ']'


class ResultStorage:
    def __init__(self, proj=PROJ_CODE, prefix=None):
        if prefix:
            self.json_dir = Path(__file__).parent / 'results' / proj / prefix
        else:
            self.json_dir = Path(__file__).parent / 'results' / proj
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.results_cache = {}
        self.attachments_name_cache = {}
        self.prefix = prefix
        self._copy = copy.deepcopy

    def save_result(self, result, job_name, overwrite=False):
        self.update_results()
        rnd_str = None
        if job_name in self.results_cache:
            if overwrite:
                old_rnd_str = random_str()
                with open(self.json_dir / f'{job_name}.result.json', mode='r') as of:
                    with open(self.json_dir / f'{job_name}.old{old_rnd_str}.result.json', mode='w') as nf:
                        nf.write(of.read())
            else:
                rnd_str = random_str()
                job_name = f'{job_name}.dup{rnd_str}'
        with open(self.json_dir / f'{job_name}.result.json', mode='w') as f:
            json.dump(result, f, cls=NpEncoder, indent=1)
        return rnd_str

    def save_attachments(self, attachments, job_name, rnd_str=None):
        for name, pkg in attachments.items():
            data = pkg['data']
            kind = pkg['kind']
            try:
                if kind == 'pickle':
                    with lzma.open(
                            self.json_dir / f'{job_name}.dup{rnd_str}.{name}.pkl.xz' if rnd_str else self.json_dir / f'{job_name}.{name}.pkl.xz',
                            mode='wb', preset=9) as f:
                        pickle.dump(data, f, protocol=5)
                elif kind == 'dill':
                    with lzma.open(
                            self.json_dir / f'{job_name}.dup{rnd_str}.{name}.dill.xz' if rnd_str else self.json_dir / f'{job_name}.{name}.dill.xz',
                            mode='wb', preset=9) as f:
                        dill.dump(data, f, protocol=5)
                elif kind == 'json':
                    with open(
                            self.json_dir / f'{job_name}.dup{rnd_str}.{name}.json' if rnd_str else self.json_dir / f'{job_name}.{name}.json',
                            mode='w') as f:
                        json.dump(data, f, cls=NpEncoder, indent=1)
                elif kind == 'pandas_csv':
                    data.to_csv(
                        self.json_dir / f'{job_name}.dup{rnd_str}.{name}.csv' if rnd_str else self.json_dir / f'{job_name}.{name}.csv',
                        index=False)
                elif kind == 'fig':
                    data.savefig(
                        self.json_dir / f'{job_name}.dup{rnd_str}.{name}.pdf' if rnd_str else self.json_dir / f'{job_name}.{name}.pdf')
                elif kind == 'weights_plot':
                    plot_mlp_model(data, input_cols=X_COLS, vmax=1).savefig(
                        self.json_dir / f'{job_name}.dup{rnd_str}.{name}.pdf' if rnd_str else self.json_dir / f'{job_name}.{name}.pdf')
                else:
                    auto_log(f'Unknown data kind "{kind}"', level='error')
            except Exception as e:
                auto_log(f'Failed to save data "{name}" ({kind}): {e}', level='error')

    def load_result(self, job_name):
        if job_name in self.results_cache:
            return self._copy(self.results_cache[job_name])
        else:
            self.update_results()
            return self._copy(self.results_cache[job_name])

    def update_results(self):
        json_files = glob.glob(str(self.json_dir / '*.result.json'))
        for fname in json_files:
            job_name = os.path.splitext(os.path.basename(fname))[0][:-len('.result')]
            if job_name not in self.results_cache:
                try:
                    with open(fname, mode='r') as f:
                        self.results_cache[job_name] = json.load(f)
                    self.attachments_name_cache[job_name] = []
                    other_files = glob.glob(str(self.json_dir / f'{job_name}.*.*'))
                    for ofname in other_files:
                        base_name = os.path.basename(ofname)
                        if ('.dup' in base_name) or ('.result' in base_name): continue
                        attachment_name = '.'.join(base_name.split('.')[-2:])
                        self.attachments_name_cache[job_name].append(attachment_name)
                except Exception as e:
                    auto_log(str(e), level='error')

    def get_all_result_names(self):
        self.update_results()
        return list(self.results_cache.keys())


def cross_validate_model(build_fn, params, X_train, y_train, cv=CV_N_FOLD, n_jobs=-1):
    # tf.config.threading.set_inter_op_parallelism_threads(2)
    # tf.config.threading.set_intra_op_parallelism_threads(2)

    model = build_fn(params)
    with parallel_backend('loky', n_jobs=n_jobs):
        cv_results = cross_validate(model, X_train, y_train, cv=KFold(n_splits=cv, shuffle=True, random_state=params.get('random_state', 0)),
                                    scoring=['neg_mean_absolute_percentage_error',
                                             'neg_mean_absolute_error',
                                             'neg_mean_squared_error'],
                                    return_train_score=True, return_estimator=True)
    return cv_results


def train_model(build_fn, params, X_train, y_train):
    # tf.config.threading.set_inter_op_parallelism_threads(2)
    # tf.config.threading.set_intra_op_parallelism_threads(2)

    model = build_fn(params)
    with parallel_backend('loky', n_jobs=-1):
        model.fit(X_train, y_train)
    return model


def objective(round_name, y_col, use_cross_validate=True, **params):
    y_scale = Y_SCALES[Y_COLS.index(y_col)]

    X_train, y_train = load_data(round_name, da_tag=params['da_tag'], y=y_col)
    X_test, y_test = load_data('test', y=y_col)

    y_col_arr = y_col.split('-')

    y_train = y_train[y_col_arr]
    y_test = y_test[y_col_arr]

    y_train /= y_scale
    y_test /= y_scale

    job_hash = obj_to_hash(params)
    job_name = f'{round_name}_{y_col}_{job_hash}'

    result = {
        'version': 'v8',
        'round_name': round_name,
        'y_col': y_col,
        'y_scale': y_scale,
        'params': dict(**params),
        'job_name': job_name,
        'hashes': {
            'params': obj_to_hash(params),
            'x_train': obj_to_hash(X_train),
            'y_train': obj_to_hash(y_train),
            'job': job_hash
        },
        'job': {
            'host': platform.node(),
            'system': platform.version(),
            'python': {
                'version': platform.python_version(),
                'implementation': platform.python_implementation(),
                'compiler': platform.python_compiler(),
            },
            'slurm': {
                'SLURM_JOB_ID': os.environ.get('SLURM_JOB_ID'),
                'SLURM_JOB_NAME': os.environ.get('SLURM_JOB_NAME'),
                'SLURM_JOB_NODELIST': os.environ.get('SLURM_JOB_NODELIST'),
                'SLURM_ARRAY_JOB_ID': os.environ.get('SLURM_ARRAY_JOB_ID'),
                'SLURM_ARRAY_TASK_ID': os.environ.get('SLURM_ARRAY_TASK_ID'),
            },
            'environ': {
                'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
                'TF_NUM_INTRAOP_THREADS': os.environ.get('TF_NUM_INTRAOP_THREADS'),
                'TF_NUM_INTEROP_THREADS': os.environ.get('TF_NUM_INTEROP_THREADS'),
            },
            'started_on': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'started_on_utc': datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            'packages': {
                'numpy': np.__version__,
                'pandas': pd.__version__,
                'tensorflow': tf.__version__,
                'tensorflow-keras': tf.keras.__version__,
                'sklearn': sklearn.__version__,
                'skopt': skopt.__version__,
                'scikeras': scikeras.__version__,
            }
        }
    }
    
    result['job']['load_average'] = os.getloadavg()

    if use_cross_validate:
        with parallel_backend('loky', n_jobs=-1):
            model, cv_results = Parallel()(delayed(f)(build_model, params, X_train, y_train) for f in (train_model, cross_validate_model))
        
        cv_models = cv_results['estimator']
        del cv_results['estimator']
        result['cv_results'] = cv_results
    else:
        model = train_model(build_model, params, X_train, y_train)
        
    result['test_errors'] = {}
    result['train_errors'] = {}
    for m in ['mean_absolute_percentage_error', 'mean_squared_error', 'mean_absolute_error']:
        try:
            result['test_errors'][m] = getattr(metrics, m)(y_test * y_scale, model.predict(X_test) * y_scale)
        except Exception as e:
            result['test_errors'][m] = -1
        try:
            result['train_errors'][m] = getattr(metrics, m)(y_train * y_scale, model.predict(X_train) * y_scale)
        except Exception as e:
            result['train_errors'][m] = -1

    result['job']['ended_on'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result['job']['ended_on_utc'] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    df_csv = load_data('all', return_df=True)[['sID'] + X_COLS]
    df_csv[[f'{yi}_Pred' for yi in y_col_arr]] = pd.DataFrame(model.predict(df_csv[X_COLS]) * y_scale)
    if use_cross_validate:
        for i, m in enumerate(cv_models):
            df_csv[[f'{yi}_PredCV{i}' for yi in y_col_arr]] = pd.DataFrame(m.predict(df_csv[X_COLS]) * y_scale)

    attachments = {'model': dict(kind='dill', data=model),
                   'weights_plot': dict(kind='weights_plot', data=model)}
    if use_cross_validate:
        attachments['model_cv'] = dict(kind='dill', data=cv_models)
    attachments['prediction'] = dict(kind='pandas_csv', data=df_csv)

    result['attachments'] = list(attachments.keys())

    return result, attachments


def hp_opt(round_name, y_col, random_seed=0, n_calls=MAX_N_MODELS_PER_Y):
    OPT_SPACE = [Integer(8, 512, name='n_top_nodes'),
                 Integer(4, 256, name='n_bot_nodes'),
                 Integer(2, 16, name='n_layers'),
                 Real(0, 1, name='linearity'),
                 Real(10 ** -7, 10 ** -2, "log-uniform", name='lr'),
                 Real(10 ** -9, 10 ** -4, "log-uniform", name='l1_coeff'),
                 Real(10 ** -9, 10 ** -4, "log-uniform", name='l2_coeff'),
                 Real(0, 0.3, name='dropout_p'),
                 Categorical(categories=[0, 1, 42, ], name='random_state'),
                 Categorical(categories=['leaky_relu', 'sigmoid', 'tanh', 'selu'], name='nonlin'),
                 Categorical(categories=AVAILABLE_DA_TAGS, name='da_tag'),
                 Categorical(categories=[True, False], name='early_stopping'),
                 Categorical(categories=['mean_squared_error'], name='loss_fn'),
                 Categorical(categories=[50000, ], name='epochs'),  # 20000
                 Categorical(categories=[4096, ], name='batch_size'),
                 Categorical(categories=[Y_NONLIN_LAST[Y_COLS.index(y_col)], ], name='nonlin_last'),
                 Categorical(categories=['normal', ], name='kernel'),
                 Categorical(categories=['none', ], name='boost'),
                 ]

    auto_log(f'HP-Opt started on round_{round_name} @ {y_col} with (random_seed={random_seed}, n_calls={n_calls}).')

    space = skopt.space.Space(OPT_SPACE)
    dimensions = skopt.utils.normalize_dimensions(space)
    rng = skopt.utils.check_random_state(random_seed)
    base_estimator = skopt.utils.cook_estimator("GP", space=dimensions,
                                                random_state=rng.randint(0, np.iinfo(np.int32).max),
                                                noise='gaussian')

    optimizer = skopt.optimizer.Optimizer(dimensions, base_estimator, 
                                          random_state=rng.randint(0, np.iinfo(np.int32).max), 
                                          n_initial_points=N_INIT_POINTS, initial_point_generator='random')
    loaded_results = []

    job_prefix = f'{round_name}_{y_col}'
    rs = ResultStorage(prefix=job_prefix)

    n_told, n_trained = 0, 0
    for i in range(n_calls):
        # Load all cached jobs
        cached_jobs = rs.get_all_result_names()
        next_xs, next_ys = [], []
        n_skip = 0
        for job_name in cached_jobs:
            if job_name.startswith(job_prefix) and job_name not in loaded_results:
                r = rs.load_result(job_name)

                next_y = (-np.mean(r['cv_results']['test_neg_mean_absolute_percentage_error']) + r['train_errors'][
                    'mean_absolute_percentage_error']) / 2.0
                next_x = [r['params'][s.name] for s in space]

                if next_x in space:
                    n_told += 1
                    next_xs.append(next_x)
                    next_ys.append(next_y)
                    # auto_log(f'Loaded saved result <{job_name}> @ {list_to_str(next_x)} = {next_y}')
                    # auto_log(f'Loaded saved result <{job_name}>')
                else:
                    n_skip += 1
                    # auto_log(f'Skipped saved result <{job_name}> @ {list_to_str(next_x)} = {next_y} (not-in-space)')
                    # auto_log(f'Skipped saved result <{job_name}> (not-in-space)')

                loaded_results.append(job_name)
        if len(next_xs) > 0:
            auto_log(f'Updating optimizer with {len(next_xs)} new points: n_told = {n_told}, n_skip = {n_skip}.')
            opt_res = optimizer.tell(next_xs, next_ys)
            opt_res.dimensions = dimensions
            opt_res.space = space

        if n_told >= n_calls:
            auto_log(f'HP-Opt ended on round_{round_name} @ {y_col}: reaching maximum results.')
            break

        next_x = optimizer.ask()
        params = {s.name: v for s, v in zip(space, next_x)}

        auto_log(f'Evaluation start @ {list_to_str(next_x)}')
        result, attachments = objective(round_name, y_col, **params)
        job_name = result['job_name']

        rnd_str = rs.save_result(result, job_name)
        rs.save_attachments(attachments, job_name, rnd_str=rnd_str)
        
        n_trained += 1
        auto_log(f'Evaluation ended <{job_name}> @ {list_to_str(next_x)}')

        if n_trained >= MAX_N_MODELS_PER_JOB:
            break

    else:
        auto_log(f'HP-Opt ended on round_{round_name} @ {y_col}: reaching maximum iterations.')


def run_job(jobs, job_idx):
    round_name, y_col, random_seed, n_calls = jobs[job_idx]
    hp_opt(round_name, y_col, random_seed=random_seed, n_calls=n_calls)


if __name__ == '__main__':
    for _ in range(2):
        jobs = get_sorted_job_list()

        _array_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        if _array_id:
            _n_jobs = len(jobs)
            _array_id = int(_array_id)
            _random_seed = random_int()
            _job = list(jobs[_array_id % _n_jobs])
            _job[2] = _random_seed
            jobs = [tuple(_job)]

        print('Job array id:', os.environ.get('SLURM_ARRAY_TASK_ID'))
        print('Jobs:', jobs)
        Parallel(n_jobs=1)(delayed(run_job)(jobs, i) for i in range(len(jobs)))

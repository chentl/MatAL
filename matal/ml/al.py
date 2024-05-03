import copy
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist, pdist
import warnings

def generate_random_compositions(materials, n_comps=30, n_sources_per_comp=None,
                                 random_state=None, return_df=True,
                                 feas_func=None, feas_factor=4):
    rng = np.random.default_rng(random_state)

    if feas_factor is None:
        feas_factor = 10 if feas_func else 1

    comps = rng.random((n_comps * feas_factor, len(materials)))

    if n_sources_per_comp is not None:
        mask_arr = [0, ] * (len(materials) - n_sources_per_comp) + [1, ] * n_sources_per_comp
        comps *= np.array([sorted(mask_arr, key=lambda _: rng.random()) for _ in range(n_comps * feas_factor)], dtype=np.float64)
    comps = comps / comps.sum(axis=1).reshape(-1, 1)
    if feas_func is not None:
        warnings.filterwarnings("ignore")
        comps = comps[feas_func(comps)]

    if return_df:
        return pd.DataFrame(comps[:n_comps], columns=materials)
    else:
        return comps[:n_comps]


class CompositionGenerator:
    '''
    Generating compositions using A-Score and Monte Carlo method.
    '''
    def __init__(self, materials=None, n_comps=30, n_iters=500000, existing_comps: pd.DataFrame = None,
                 random_state=None, perf_func=None, **kwargs):
        self.materials = materials
        self.n_comps = n_comps
        self.existing_comps = existing_comps
        self.n_iters = n_iters
        self.random_state = random_state
        self.perf_func = perf_func
        self.kwargs = kwargs
        self._rng = np.random.default_rng(random_state)

    def _get_rand_comp(self, with_pred=False):
        ''' return a random composition '''

        # If the cache does not exist, or the cache is empty, rebuild cache
        comp_cache = getattr(self, '_comp_cache', [])
        if len(comp_cache) == 0:
            self._build_comp_cache()

        # pop one composition, and with its prediction if existed, from cache
        comp = self._comp_cache.pop()
        pred = self._pred_cache.pop() if self.perf_func else None
        if with_pred:
            return comp, pred
        else:
            return comp

    def _build_comp_cache(self):
        ''' build cache containing a list of random composition '''
        seed = self._rng.integers(0, np.iinfo(np.int64).max)
        comps = generate_random_compositions(self.materials, n_comps=self.n_comps * 10,
                                             random_state=seed, return_df=False, **self.kwargs)
        self._comp_cache = [c for c in comps]
        if self.perf_func:
            self._pred_cache = [c for c in self.perf_func(self._comp_cache)]

    def optimize(self):
        # global comps, idx
        # Initialize comps and calculate its score
        best_comps = [self._get_rand_comp() for _ in range(self.n_comps)]
        best_score = self.score_comps(best_comps, perf_func=self.perf_func)

        if self.perf_func:
            # Keep a list of prediction so we don't need to re-predict all of them
            # each time we changed one composition
            best_predictions = self.perf_func(best_comps)

        for i in tqdm(range(self.n_iters), leave=False):
            if self.perf_func:
                # Randomly select a composition, compositions with lower
                # predicted performance are more likely to be selected.
                sorted_idxs = np.argsort(best_predictions)
                p = np.e ** (-1.0 * np.arange(len(sorted_idxs)))
                idx = self._rng.choice(sorted_idxs, p=p / p.sum())[0]
            else:
                # Randomly select a composition
                idx = self._rng.integers(0, len(best_comps))

            # Replace the composition with a new random one
            comps = copy.deepcopy(best_comps)
            new_comp, new_pred = self._get_rand_comp(with_pred=True)
            comps[idx] = new_comp

            # Update predictions
            if self.perf_func:
                predictions = copy.deepcopy(best_predictions)
                predictions[idx] = new_pred
            else:
                predictions = None

            # Update score
            score = self.score_comps(comps, predictions=predictions)

            if score > best_score:
                best_comps = comps
                best_score = score
                if self.perf_func:
                    best_predictions = predictions

        return pd.DataFrame(best_comps, columns=self.materials)

    def score_comps(self, comps, perf_func=None, predictions=None):
        if predictions is None:
            if perf_func is None:
                predictions = [1.0]
            else:
                predictions = perf_func(comps)
        perf_score = np.mean(predictions)

        min_intra_distance = pdist(comps).min()

        if self.existing_comps is not None:
            min_inter_distance = cdist(self.existing_comps, comps).min()
        else:
            min_inter_distance = np.inf

        distance = min(min_intra_distance, min_inter_distance)

        # score of a list of comps = (min paired-distance) * mean(predicted performance)
        score = distance * perf_score

        return score
from dataclasses import dataclass
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from dataset_engineering.source import compute_envs

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 800


@dataclass
class Set:
    idx: int
    start: int
    end: int


@dataclass
class Walk:
    train: Set
    val: Set
    test: Set


class WalkForwardFull:
    def __init__(self,
                 data: pd.DataFrame,
                 start_test=pd.to_datetime('2024-07-01'),
                 val_size: int = 20,  # 20 * 3,
                 test_size: int = 20,
                 gap: int = 0,
                 fix_start: bool = False):
        self.val_size = val_size
        self.test_size = test_size
        self.gap = gap + 1
        self.dates = pd.to_datetime(data.index)
        self.train_size = list(self.dates).index(start_test) - (self.val_size + self.gap + 1)
        self.fix_start = fix_start
        self.count = 0

    def split(self):
        i, start_train, breakk = 0, 0, False
        while True:
            if not self.fix_start:
                if i == 0:
                    end_train = start_train + self.train_size
                elif i >= 1:
                    end_train += self.val_size + 1
            else:
                pass

            start_val = end_train + self.gap
            stop_val = start_val + self.val_size
            start_test = stop_val + self.gap
            stop_test = start_test + self.test_size

            if stop_test >= len(self.dates) - self.gap:
                stop_test = len(self.dates) - self.gap - 1
                breakk = True

            if not self.fix_start:
                if i == 1:
                    start_train = self.train_size + 1
                elif i > 1:
                    start_train += self.val_size + 1

            yield (start_train, end_train), (start_val, stop_val), (start_test, stop_test)
            i += 1
            self.count = i
            if breakk:
                break

    def get_walks(self, verbose: bool = True):
        idx = 0
        date = pd.DataFrame(index=self.dates)
        for train_index, valid_index, test_index in self.split():
            idx += 1
            if not self.fix_start:
                start_train = self.dates[train_index[0]] if idx == 1 else self.dates[train_index[0]]
            else:
                start_train = self.dates[train_index[0]]

            end_train = self.dates[train_index[-1]]
            start_valid = self.dates[valid_index[0]]
            end_valid = self.dates[valid_index[-1]]
            start_test = self.dates[test_index[0]]
            end_test = self.dates[test_index[-1]]

            walk = Walk(train=Set(idx=idx, start=start_train, end=end_train),
                        val=Set(idx=idx, start=start_valid, end=end_valid),
                        test=Set(idx=idx, start=start_test, end=end_test))

            if verbose:
                print('*' * 20, f'{idx}th walking forward', '*' * 20)
                print(f'Training: {walk.train.start} to {walk.train.end}, len={len(date.loc[start_train:end_train])}')
                print(f'Validation: {walk.val.start} to {walk.val.end}, len={len(date.loc[start_valid:end_valid])}')
                print(f'Test: {walk.test.start} to {walk.test.end}, len={len(date.loc[start_test:end_test])}')

            yield idx, walk


def plot_tscv(tscv, data, target, params, features):
    for i, walk in tscv.get_walks(False):
        pass

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.tick_params(labelleft=False, left=False, labelbottom=False)
    gs = fig.add_gridspec(i, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True, )
    fig.suptitle(f'Time Serie Cross Validation using days_ahead={params.days_ahead} and gap={tscv.gap}')

    for i, walk in tscv.get_walks(verbose=False):
        cv_data = pd.DataFrame(columns=['train', 'val', 'test'], index=pd.to_datetime(data.index))
        train_dates, val_dates, test_dates = walk.train, walk.val, walk.test

        train_env, val_env, test_env = compute_envs(data, features, train_dates, val_dates, test_dates,
                                                    target, params.days_ahead, scale_target=False,
                                                    n_steps=params.n_steps,
                                                    method_aggregate_target=params.method_aggregate_target)

        train_ts = pd.Series(train_env.y.squeeze(), index=train_env.dates[params.n_steps - 1:])
        val_ts = pd.Series(val_env.y.squeeze(), index=val_env.dates[params.n_steps - 1:])
        test_ts = pd.Series(test_env.y.squeeze(), index=test_env.dates[params.n_steps - 1:])

        cv_data['train'].loc[train_ts.index] = train_ts
        cv_data['val'].loc[val_ts.index] = val_ts
        cv_data['test'].loc[test_ts.index] = test_ts

        print('*' * 20, f'{i}th walking forward', '*' * 20)
        print(f'Training: {train_ts.index[0]} to {train_ts.index[-1]}, len={len(train_ts)}')
        print(f'Validation: {val_ts.index[0]} to {val_ts.index[-1]}, len={len(val_ts)}')
        print(f'Test: {test_ts.index[0]} to {test_ts.index[-1]}, len={len(test_ts)}')

        axs[i - 1].plot(cv_data)
        axs[i - 1].set_ylabel(f'Fold {i}')

    ax.legend([Patch(color='red'), Patch(color='blue'), Patch(color='purple')],
              ['Train', 'Val', 'Test'], loc=(1.02, .8))

    for ax in axs:
        ax.label_outer()

    plt.savefig(f'plots/tscv_fixstart={tscv.fix_start}.jpg')
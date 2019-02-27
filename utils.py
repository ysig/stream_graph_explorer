import datetime
import pandas as pd
import numpy as np
from sklearn.datasets.lfw import Bunch
from six import iteritems
from datetime import datetime, timedelta
from sklearn.datasets.lfw import Bunch

# Wrapper for handling categories (stream-graphs as dictionary values)
def categorical(func):
    def call(obj, *args, **kwargs):
        if isinstance(obj, dict) and not isinstance(obj, Bunch):
            return {c: func(obj_c, *args, **kwargs) for c, obj_c in iteritems(obj)}
        else:
            return func(obj, *args, **kwargs)
    return call

# utils
@categorical
def unravel_time(data, flip=False):
    # First dimension is considered as time
    if flip:
        return {(u, t): v for t, d in iteritems(data) for u, v in iteritems(d)}
    else:
        return {(t, u): v for t, d in iteritems(data) for u, v in iteritems(d)}


def time_discretizer(df, step, inplace=False):
    assert (isinstance(step, int) or isinstance(step, timedelta)) and isinstance(df, pd.DataFrame)
    if not inplace:
        df = df.copy()
    if isinstance(step, timedelta):
        step = step.total_seconds()
    time_min, time_max = df.ts.min(), df.ts.max()
    bins = np.arange(time_min, time_max, step).tolist()
    if time_max != bins[-1]:
        bins.append(time_max)
    if len(bins) <= 1:
        raise ValueError('please provide a bigger bin size')
    df["ts"] = pd.cut(df["ts"], bins, labels=list(range(len(bins) - 1)), include_lowest=True)
    return df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'w'}), bins


def make_minimal_stream_graph(df):
    columns = set(df.columns)
    ndf = df[list(columns - {'v', 'w'})].append(df[list(columns - {'u', 'w'})].rename(columns={'v': 'u'}), ignore_index=True, sort=False).drop_duplicates()    
    columns = set(ndf.columns)
    ns, ts = df[list(columns - {'ts'})].drop_duplicates(), df[list(columns - {'u'})].drop_duplicates()
    return Bunch(nodeset=ns, timeset=ts, nodestream=ndf, linkstream=df)

def make_bin_map(bins):
    return {i: [datetime.fromtimestamp(s).strftime("%Y-%m-%d %H:%M"), datetime.fromtimestamp(f).strftime("%Y-%m-%d %H:%M")] for (i, (s, f)) in enumerate(zip(bins[:-1], bins[1:]))}

def categorical_split(df, category_name='c'):
    if category_name in df.columns:
        return {k: sdf.drop(columns=category_name) for k, sdf in df.groupby([category_name])}
    else:
        return df

if __name__=="__main__":
    df, step = pd.DataFrame([[0, 1, 1], [0, 1, 2], [1, 2, 3], [2, 1, 4]], columns=['u', 'v', 'ts']), 2
    print(time_discretizer(df, step))

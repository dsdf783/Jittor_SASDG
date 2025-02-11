import jittor as jt
import glob
import os
import re
from pathlib import Path
from dataset.scaler import MinMaxScaler


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    path = Path(path)
    if path.exists() and (not exist_ok):
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f'{path}{sep}*')
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = Path(f'{path}{sep}{n}{suffix}')
    dir = path if path.suffix == "" else path.parent
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path

class Normalizer():

    def __init__(self, data):
        flat = data.reshape(-1, data.shape[-1])
        self.scaler = MinMaxScaler((- 1, 1), clip=True)
        self.scaler.fit(flat)

    def normalize(self, x):
        batch, seq, ch = x.shape
        x = x.reshape(-1, ch)
        return self.scaler.transform(x).reshape((batch, seq, ch))

    def unnormalize(self, x):
        batch, seq, ch = x.shape
        x = x.reshape(-1, ch)
        x = jt.array(x) 
        x = jt.clamp(x, min_v=-1, max_v=1)
        x=x.numpy()
        return self.scaler.inverse_transform(x).reshape((batch, seq, ch))

def vectorize_many(data):
    batch_size = data[0].shape[0]
    seq_len = data[0].shape[1]
    out = [jt.array(x).reshape(batch_size, seq_len, -1).contiguous() for x in data]

    global_pose_vec_gt = jt.concat(out, dim=2)
    return global_pose_vec_gt




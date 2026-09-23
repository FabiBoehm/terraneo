"""Dataset loading and the two losses shared by the training and evaluation code.
Extracted from the retired train_operator.py; the linear-operator trainer is
the only remaining user."""
import os

import numpy as np
import torch


def load_coords(root):
    """Node positions from stokes_dataset_tool --dump-coords: (S, nx, ny, nr, 3)."""
    import json
    import os

    root = os.path.expanduser(root)
    mesh = json.load(open(os.path.join(root, "mesh.json")))
    shape = tuple(mesh["velocity_shape"])
    return np.fromfile(os.path.join(root, "coords_velocity.bin"),
                       dtype=np.float64).reshape(*shape, 3)


def load_coords_pressure(root):
    """Node positions of the coarse pressure grid: (S, nx, ny, nr, 3)."""
    import json
    import os

    root = os.path.expanduser(root)
    mesh = json.load(open(os.path.join(root, "mesh.json")))
    shape = tuple(mesh["pressure_shape"])
    return np.fromfile(os.path.join(root, "coords_pressure.bin"),
                       dtype=np.float64).reshape(*shape, 3)


def load_split(root, split, verbose=True, limit=None):
    """Loads a split as whole-shell samples: (n_samples, n_subdomains, nx, ny, nr, C).

    ``limit`` truncates the split. At level 5 a full split is ~40 GB of raw arrays
    before the derived channels, so being able to cap it is what makes the finer
    meshes trainable at all on one node.
    """
    from terra_data.dataset import StokesDataset

    # One .npz per (split, limit) next to the data: reading 1600 L5 samples as
    # individual 13 MB files takes 1-3 h on the shared filesystem (and several
    # concurrent runs make it worse), while the packed archive is one sequential
    # read. TERRA_NO_SPLIT_CACHE=1 disables it.
    cache = os.path.join(root, f"_cache_{split}_{limit if limit is not None else 'all'}.npz")
    if os.environ.get("TERRA_NO_SPLIT_CACHE", "0") != "1" and os.path.exists(cache):
        try:
            with np.load(cache, allow_pickle=False) as z_:
                d = {k: z_[k] for k in z_.files}
            d["z"] = d["z"] if "z" in d and d["z"].ndim > 1 else None
            if verbose:
                print(f"  {split}: {len(d['f_u'])} whole-shell samples of "
                      f"{d['f_u'].shape[1:]} (from cache)", flush=True)
            return d
        except Exception as e:                       # corrupt/partial cache: rebuild
            print(f"  cache {cache} unreadable ({str(e)[:60]}), rebuilding", flush=True)

    ds = StokesDataset(root, split)
    if limit is not None:
        ds = _Truncated(ds, limit)
    f_u = np.empty((len(ds), *ds.shapes["u"]), dtype=np.float32)
    u = np.empty_like(f_u)
    p = np.empty((len(ds), *ds.shapes["p_fine"]), dtype=np.float32)
    f_p = np.empty((len(ds), *ds.shapes["f_p"]), dtype=np.float32)
    f_p_v = np.empty((len(ds), *ds.shapes["f_p_fine"]), dtype=np.float32)
    log_eta = np.empty((len(ds), *ds.shapes["eta"]), dtype=np.float32)
    eta_mean = np.empty(len(ds), dtype=np.float32)
    z = (np.empty((len(ds), *ds.shapes["z"]), dtype=np.float32)
         if "z" in ds.shapes else None)

    for i in range(len(ds)):
        smp = ds[i]
        if z is not None:
            z[i] = smp["z"]
        f_u[i] = smp["f_u"]
        u[i] = smp["u"]
        p[i] = smp["p_fine"]
        f_p[i] = smp["f_p"]
        f_p_v[i] = smp["f_p_fine"]
        le = np.log(np.maximum(smp["eta"], 1e-12))
        log_eta[i] = le
        # One scale per sample -- this is the granularity at which the -0.985
        # correlation between log rms(u) and log(mean eta) was measured.
        eta_mean[i] = np.exp(le.mean())

    if verbose:
        print(f"  {split}: {len(ds)} whole-shell samples of {f_u.shape[1:]}")
    out = dict(f_u=f_u, u=u, p=p, f_p=f_p, f_p_v=f_p_v,
               log_eta=log_eta, eta_mean=eta_mean, z=z)
    if os.environ.get("TERRA_NO_SPLIT_CACHE", "0") != "1" and os.access(root, os.W_OK):
        # np.savez appends .npz when the name lacks it, so the temp name must already
        # end in .npz or the rename below looks for a file that was never written.
        tmp = cache + f".tmp{os.getpid()}.npz"       # atomic: concurrent runs are safe
        try:
            np.savez(tmp, **{k: v for k, v in out.items() if v is not None})
            os.replace(tmp, cache)
            print(f"  cached -> {os.path.basename(cache)}", flush=True)
        except Exception as e:
            print(f"  cache write failed ({str(e)[:60]})", flush=True)
            if os.path.exists(tmp):
                os.remove(tmp)
    return out


class _Truncated:
    """A view of the first ``n`` samples of a dataset."""

    def __init__(self, ds, n):
        self._ds, self._n = ds, min(n, len(ds))
        self.shapes = ds.shapes

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return self._ds[i]


def mean_free(x):
    """Removes the per-sample constant. A constant in p leaves grad p unchanged, so the
    physics does not determine it and the network should not be scored on it."""
    return x - x.mean(dim=(1, 2, 3, 4, 5), keepdim=True)


def relative_l2(pred, target, eps=1e-12):
    """Per-sample ||pred - target|| / ||target||, averaged over the batch."""
    return relative_l2_ps(pred, target, eps).mean()


def relative_l2_ps(pred, target, eps=1e-12):
    """Per-sample ||pred - target|| / ||target||, NOT averaged: (batch,)."""
    dims = tuple(range(1, pred.ndim))
    num = torch.sqrt(torch.sum((pred - target) ** 2, dim=dims))
    den = torch.sqrt(torch.sum(target**2, dim=dims)) + eps
    return num / den

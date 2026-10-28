'''
Loaders for the data products written by the `summarize_*` scripts.

Each module here mirrors the `summarize_*` module that writes the files it
reads, and owns the `SUFFIXES` mapping that names those files -- the
`summarize_*` module imports that mapping back for its expected-suffix list so
that the reader and writer cannot drift apart.

Loaders are accessed through the qualified path, which says which experiment
the file came from:

    from cftsdata import loaders
    loaders.abr.waveforms(path)
    loaders.efr.harmonics(path)
    loaders.dpoae.th(path)

Every loader accepts either the dataset folder or the data product itself.
Given the folder, the expected filename is derived from `SUFFIXES`. The
decorator that does so is re-exported here as `loaders.ensure_suffix` for
writing your own loaders; `cftsdata.util` has the filename parser and the
helpers for reshaping what the loaders return.
'''
from ._util import ensure_suffix
from . import abr, dpoae, efr, memr, mlr_llr

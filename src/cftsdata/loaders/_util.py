'''
Plumbing shared by the loaders in this package.

Exposed as `cftsdata.loaders.ensure_suffix` for decorating ad-hoc loaders.
'''
from functools import wraps
from pathlib import Path


def ensure_suffix(default_suffix):
    '''
    Allow a loader to be called with either the dataset folder or the file

    The summarize scripts save each data product to the dataset folder as
    `{dataset stem} {suffix}`, so given the folder we can map straight to the
    expected filename.

    The wrapped function gains a `suffix` keyword that overrides the default.
    This is needed where the same data product is saved under more than one
    name (e.g., the Verhulst EFR analysis appends ` linear`). The default is
    also exposed as the `suffix` attribute of the wrapped function.
    '''
    def decorator(f):
        @wraps(f)
        def wrapper(filename, *args, suffix=None, **kwargs):
            if suffix is None:
                suffix = default_suffix
            filename = Path(filename)
            if not filename.name.endswith(suffix):
                filename = filename / f'{filename.stem} {suffix}'
            return f(filename, *args, **kwargs)
        wrapper.suffix = default_suffix
        return wrapper
    return decorator

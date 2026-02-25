import string
import numpy as np


def quantize_matrices(
    dataset, bmat="bmatrix", qmat="qmatrix", assert_number=0, round=-2
):
    """quantize b-matrices"""
    for name in dataset:
        if not bmat in dataset[name]:
            continue
        elif not isinstance(dataset[name][bmat], (list, tuple)):
            dataset[name][qmat] = (0,) * 6
            continue
        dataset[name][qmat] = tuple(np.round(dataset[name][bmat], round).astype(int))
    qmats = sorted({item[qmat] for item in dataset.values() if qmat in item})
    if assert_number:
        assert len(qmats) == assert_number
    return dataset


# utilities


class NanoDB(dict):
    """Minimal database-type query using 1-nested dictionary structure

    # init
    db = NanoDB(some_dict)

    # filter
    db(db.key==value) -> new NanoDB dictionary
    db.groupby(key) -> (key1, NanoDB(group1), ...

    """

    class Selector:
        def __init__(self, field):
            self.field = field

        def __eq__(self, value):
            return lambda item: item.get(self.field) == value

        def __ne__(self, value):
            return lambda item: item.get(self.field) != value

        def is_in(self, value):
            return lambda item: item.get(self.field) in value

        def not_in(self, value):
            return lambda item: not item.get(self.field) in value

    def __getattr__(self, name):
        """return selector"""
        try:
            return super(dict, self).__getattr__(name)
        except AttributeError:
            return self.Selector(name)

    def __call__(self, *queries):
        """apply filter"""
        return NanoDB(
            {
                key: value
                for key, value in self.items()
                if all(query(value) for query in queries)
            }
        )

    def first(self):
        if not self:
            raise ValueError("Empty dataset")
        return next(iter(self))
    
    def single(self):
        if len(self) != 1:
            raise ValueError("Non singleton dataset")
        return self.first()

    def unique(self, *fields):
        """unique values for given fields"""
        if len(fields) == 1:
            return sorted(
                set(value[fields[0]] for value in self.values() if fields[0] in value)
            )
        return sorted(
            set(
                tuple(value[field] for field in fields)
                for value in self.values()
                if all(field in value for field in fields)
            )
        )

    def groupby(self, *fields):
        for key in self.unique(*fields):
            keys = (key,) if len(fields) == 1 else key
            yield key, self(
                *tuple(self.Selector(field) == key for key, field in zip(keys, fields))
            )


class Formatter:
    """convert tuple/dict of value to str"""

    def __init__(self, fmt):
        items = list(string.Formatter().parse(fmt))
        keys = [item[1] for item in items if item[1]]
        if len(keys) != len(items):
            raise ValueError("All format fields must have a field name")
        self.fmt = fmt
        self.keys = keys

    def __repr__(self):
        return self.fmt

    def __call__(self, *args, **kwargs):
        if args:
            kwargs = {**kwargs, **dict(zip(self.keys, args))}
        return self.fmt.format(**kwargs)

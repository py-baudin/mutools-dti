"""unittest for mutools.dti.utils"""
from mutools_dti import utils
import pytest


def test_nanodb_class():
    NanoDB = utils.NanoDB

    db = NanoDB({"id1": {"a": "foo", "b": "bar"}, "id2": {"a": "foo", "b": "baz"}})

    # standard dict methods
    assert db["id1"] == {"a": "foo", "b": "bar"}

    # filter query
    assert db(db.a == "foo") == db
    assert db(db.b == "bar") == {"id1": {"a": "foo", "b": "bar"}}
    assert db(db.b == "baz") == {"id2": {"a": "foo", "b": "baz"}}

    # first key
    assert db(db.a == "foo").first() == "id1"
    assert db(db.b == "baz").first() == "id2"
    with pytest.raises(ValueError):
        db.first(unique=True)

    # unique values
    assert db.unique("a") == ["foo"]

    assert dict(db.groupby("a")) == {
        "foo": {"id1": {"a": "foo", "b": "bar"}, "id2": {"a": "foo", "b": "baz"}}
    }
    assert dict(db.groupby("b")) == {
        "bar": {"id1": {"a": "foo", "b": "bar"}},
        "baz": {"id2": {"a": "foo", "b": "baz"}},
    }
    assert dict(db.groupby("a", "b")) == {
        ("foo", "bar"): {"id1": {"a": "foo", "b": "bar"}},
        ("foo", "baz"): {"id2": {"a": "foo", "b": "baz"}},
    }

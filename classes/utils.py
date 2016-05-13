
from re import sub

def toString(model):
    """Given a model and a feature set, return a short string that will serve
    as identifier for this combination.
    Ex: (LogisticRegression(), "basic_s") -> "LR:basic_s"
    """
    return "%s" % (sub("[a-z]", '', model.__class__.__name__))


from re import sub

def toString(model):
    """Given a model and a feature set, return a short string that will serve
    as identifier for this combination.
    Ex: (LogisticRegression(), "basic_s") -> "LR:basic_s"
    """
    return "%s" % (sub("[a-z]", '', model.__class__.__name__))

def plot_roc(fpr, tpr):
    """Plot ROC curve and display it."""
    plt.clf()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
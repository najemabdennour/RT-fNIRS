# core/decoders.py
"""Registry of scikit-learn classifiers available as decoders, and a factory to build them."""
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

RANDOM_STATE = 12345
# Decoder key -> factory returning a fresh, unfitted classifier
DECODER_REGISTRY = {
    "svm": lambda: CalibratedClassifierCV(SVC(class_weight='balanced', random_state=RANDOM_STATE), ensemble=False),
    "svmlinear": lambda: LinearSVC(dual=True, C=1, class_weight='balanced', random_state=RANDOM_STATE),
    "decisiontree": lambda: DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=RANDOM_STATE),
    "extratree": lambda: ExtraTreeClassifier(max_depth=15, class_weight='balanced', random_state=RANDOM_STATE),
    "randomforest": lambda: RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=RANDOM_STATE),
    "extratrees": lambda: ExtraTreesClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=RANDOM_STATE),
    "bagging": lambda: BaggingClassifier(n_estimators=50, random_state=RANDOM_STATE),
    "gradientboosting": lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE),
    "adaboost": lambda: AdaBoostClassifier(random_state=RANDOM_STATE),
    "naivebayes": lambda: GaussianNB(),
    "kneighbors": lambda: KNeighborsClassifier(),
    "mlp": lambda: MLPClassifier(max_iter=500, random_state=RANDOM_STATE),
    "sgd": lambda: SGDClassifier(random_state=RANDOM_STATE),
    "logisticregression": lambda: LogisticRegression(solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
}


def decoder_selection(preprocessing=False, selected_method="extratrees"):
    """
    Builds a fresh, unfitted decoder for a DECODER_REGISTRY key.

    With preprocessing=True, returns a Pipeline of VarianceThreshold followed
    by the classifier. Otherwise returns the bare classifier, except for
    'extratrees', which is wrapped in a single-step Pipeline. Feature scaling
    is not done here; it belongs to the offline pipeline's 'scaling' step.

    Raises ValueError for an unknown key.
    """
    if selected_method not in DECODER_REGISTRY:
        raise ValueError(f"Selected decoder method key '{selected_method}' is not registered.")

    classifier = DECODER_REGISTRY[selected_method]()

    if preprocessing:
        return make_pipeline(VarianceThreshold(), classifier)

    if selected_method == "extratrees":
        return make_pipeline(ExtraTreesClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=RANDOM_STATE))

    return classifier
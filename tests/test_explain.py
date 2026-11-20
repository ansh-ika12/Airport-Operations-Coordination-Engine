
import pytest

from config import ALL_FEATURES

FEATS = ["month", "dep_hour", "distance", "sched_duration"]


@pytest.fixture(scope="module")
def toy_model(toy_df):
    lgb = pytest.importorskip("lightgbm")
    X = toy_df[FEATS]
    clf = lgb.LGBMClassifier(n_estimators=20, num_leaves=7, verbose=-1)
    clf.fit(X, toy_df["delayed"])
    return clf, X

def test_build_explainer(toy_model):
    ex = pytest.importorskip("ml.explain")
    np = pytest.importorskip("numpy")
    clf, X = toy_model
    explainer = ex.build_explainer(clf)
    assert explainer.backend in ("shap", "lightgbm")
    vals = np.asarray(explainer.shap_values(X.iloc[:50]))
    assert vals.shape == (50, len(X.columns))      # rows x features, delayed class


def test_top_factors_shape(toy_model):
    ex = pytest.importorskip("ml.explain")
    clf, X = toy_model
    explainer = ex.build_explainer(clf)
    row = X.iloc[[0]]                              # one-row DataFrame
    out = ex.top_factors(explainer, row, row.iloc[0].to_dict(), n=3)
    assert len(out) == 3
    for item in out:
        assert {"factor", "value", "effect", "impact"} <= set(item)
        assert item["effect"] in ("increases delay risk", "reduces delay risk")


def test_global_importance_sorted(toy_model):
    ex = pytest.importorskip("ml.explain")
    clf, X = toy_model
    imp = ex.global_importance(ex.build_explainer(clf), X.iloc[:200], FEATS)
    assert len(imp) == len(FEATS)
    assert imp.is_monotonic_decreasing


def test_every_feature_has_readable_label():
    ex = pytest.importorskip("ml.explain")
    for name in ALL_FEATURES:
        label = ex.feature_label(name)
        assert isinstance(label, str) and label.strip(), name
        assert label != name, f"'{name}' has no readable label"
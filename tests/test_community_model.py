from ml.community_model import evaluate_community_model


def test_returns_documented_reason(toy_df):
    out = evaluate_community_model(toy_df)
    assert out["comparable"] is False
    assert isinstance(out["reason"], str) and len(out["reason"]) > 20
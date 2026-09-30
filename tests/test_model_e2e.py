"""End-to-end test: train the model on the synthetic dataset and verify it
genuinely beats a trivial baseline on held-out data."""

from phishing_detector.model import train_model


def test_training_pipeline_beats_trivial_baseline():
    clf, metrics, df = train_model(n_per_class=400, seed=7, test_size=0.25, n_estimators=200)

    # Trivial baseline: always predict the majority class -> ~50% accuracy
    # since the synthetic dataset is balanced. Require the real model to do
    # substantially better than that, and to clear a solid absolute bar.
    assert metrics.accuracy > 0.85, f"accuracy too low: {metrics.accuracy}"
    assert metrics.precision > 0.80
    assert metrics.recall > 0.80
    assert metrics.f1 > 0.80

    # Sanity on confusion matrix shape
    assert len(metrics.confusion_matrix) == 2
    assert len(metrics.confusion_matrix[0]) == 2

    # Dataset should be balanced-ish and reasonably sized
    assert len(df) >= 700


def test_predict_url_on_obvious_examples():
    from phishing_detector.model import predict_url

    clf, metrics, df = train_model(n_per_class=500, seed=11, test_size=0.2, n_estimators=250)

    legit_result = predict_url("https://www.google.com/search?q=test", model=clf, feature_names=list(df.columns[:-2]))
    phish_result = predict_url("http://192.168.5.23/paypal/login.php", model=clf, feature_names=list(df.columns[:-2]))

    assert legit_result["prediction"] == "legitimate"
    assert phish_result["prediction"] == "phishing"

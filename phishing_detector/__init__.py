"""Phishing URL/Email Detector.

A real, working phishing detection toolkit combining:
  - Lexical/URL feature extraction (features.py)
  - A synthetic-but-realistic dataset generator, plus an external CSV loader
    for real UCI/Kaggle data (data.py)
  - A scikit-learn training/evaluation pipeline (model.py)
  - A rule-based email phishing heuristic scorer (email_heuristics.py)
"""

__version__ = "0.1.0"

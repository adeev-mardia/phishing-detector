from phishing_detector.email_heuristics import analyze_email

PHISHING_EMAIL = """From: PayPal Security <security@paypa1-alerts.tk>
Reply-To: attacker@totally-different-domain.ru
Subject: Urgent: Your account has been limited - Action Required

Dear Customer,

We have detected unusual activity on your account. Your account will be
closed within 24 hours unless you verify your identity immediately.

Please confirm your password and credit card number by clicking below:

<a href="http://192.168.44.10/paypal/login.php">https://www.paypal.com/login</a>
<a href="http://bit.ly/xk29zA">Click here to verify</a>
<a href="http://paypal.secure-verification.tk/update">Update billing info</a>
<a href="http://paypal.secure-verification.tk/confirm">Confirm now</a>
<a href="http://paypal.secure-verification.tk/account">My account</a>
<a href="http://paypal.secure-verification.tk/help">Help</a>

Act now to avoid permanent suspension.
"""

LEGITIMATE_EMAIL = """From: GitHub <noreply@github.com>
Subject: [GitHub] A new SSH key was added to your account

Hi adeev-mardia,

A new SSH key was recently added to your account. If you added this key,
you don't need to do anything. If you did not add this key, please review
your account security settings.

View your SSH keys: https://github.com/settings/keys

Thanks,
The GitHub Team
"""


def test_phishing_email_is_flagged_phishing():
    result = analyze_email(raw_email=PHISHING_EMAIL)
    assert result.verdict in ("phishing", "suspicious")
    assert result.score >= 50
    signal_names = {s.name for s in result.signals}
    assert "urgency_language" in signal_names
    assert "generic_greeting_plus_credential_request" in signal_names
    assert "sender_domain_brand_mismatch" in signal_names


def test_legitimate_email_is_not_flagged():
    result = analyze_email(raw_email=LEGITIMATE_EMAIL)
    assert result.verdict == "legitimate"
    assert result.score < 20


def test_link_text_href_mismatch_detected():
    result = analyze_email(raw_email=PHISHING_EMAIL)
    signal_names = {s.name for s in result.signals}
    assert "link_text_href_mismatch" in signal_names


def test_reply_to_mismatch_detected():
    result = analyze_email(raw_email=PHISHING_EMAIL)
    signal_names = {s.name for s in result.signals}
    assert "reply_to_mismatch" in signal_names


def test_field_based_call_without_raw_email():
    result = analyze_email(
        subject="Urgent: verify your account",
        sender="Support <support@example-secure-login.tk>",
        body="Dear Customer, please confirm your password and credit card number immediately. Act now.",
    )
    assert result.score > 0
    assert result.verdict in ("suspicious", "phishing")

import pytest

from phishing_detector.features import (
    FEATURE_NAMES,
    brand_name_mismatch,
    digit_ratio,
    extract_features,
    has_at_symbol,
    has_double_slash_redirect,
    has_ip_address,
    has_non_standard_port_for_scheme,
    has_port_in_url,
    has_sensitive_words_in_path,
    has_suspicious_tld,
    hostname_length,
    hostname_shannon_entropy,
    https_token_in_domain,
    is_shortened_url,
    longest_word_length_in_hostname,
    num_dots,
    num_hyphens_in_domain,
    num_query_params,
    num_special_chars,
    num_subdomains,
    path_depth,
    url_length,
    uses_https,
)


def test_has_ip_address_true_for_ipv4_host():
    assert has_ip_address("http://192.168.1.1/login") == 1


def test_has_ip_address_false_for_domain():
    assert has_ip_address("https://www.google.com/") == 0


def test_has_ip_address_true_for_hex_ip():
    assert has_ip_address("http://0x1A2B3C4D/login") == 1


def test_has_at_symbol_true():
    assert has_at_symbol("http://google.com@evil.com/login") == 1


def test_has_at_symbol_false():
    assert has_at_symbol("http://google.com/login") == 0


def test_num_dots_counts_hostname_dots_only():
    assert num_dots("https://a.b.c.example.com/x.y.z") == 4


def test_num_subdomains():
    assert num_subdomains("https://a.b.example.com/") == 2
    assert num_subdomains("https://example.com/") == 0
    assert num_subdomains("https://www.example.com/") == 1


def test_uses_https():
    assert uses_https("https://example.com") == 1
    assert uses_https("http://example.com") == 0


def test_https_token_in_domain():
    assert https_token_in_domain("http://https-paypal-secure.com/login") == 1
    assert https_token_in_domain("https://paypal.com/login") == 0


def test_num_hyphens_in_domain():
    assert num_hyphens_in_domain("http://pay-pal-secure-login.com") == 3
    assert num_hyphens_in_domain("http://paypal.com") == 0


def test_digit_ratio():
    assert digit_ratio("") == 0.0
    ratio = digit_ratio("http://12345.com")
    assert 0 < ratio < 1


def test_is_shortened_url():
    assert is_shortened_url("http://bit.ly/abc123") == 1
    assert is_shortened_url("http://example.com/abc123") == 0


def test_has_suspicious_tld():
    assert has_suspicious_tld("http://free-gift.top/claim") == 1
    assert has_suspicious_tld("http://example.com/claim") == 0


def test_path_depth():
    assert path_depth("http://example.com/a/b/c") == 3
    assert path_depth("http://example.com/") == 0


def test_num_query_params():
    assert num_query_params("http://example.com/?a=1&b=2") == 2
    assert num_query_params("http://example.com/") == 0


def test_has_double_slash_redirect():
    assert has_double_slash_redirect("http://legit.com//evil.com") == 1
    assert has_double_slash_redirect("http://legit.com/path") == 0


def test_num_special_chars():
    assert num_special_chars("http://example.com/a;b,c") >= 2


def test_brand_name_mismatch_true_for_brand_outside_domain():
    assert brand_name_mismatch("http://paypal.secure-login.com/") == 1


def test_brand_name_mismatch_false_for_real_domain():
    assert brand_name_mismatch("https://www.paypal.com/signin") == 0


def test_has_port_in_url():
    assert has_port_in_url("http://example.com:8080/") == 1
    assert has_port_in_url("http://example.com/") == 0


def test_has_non_standard_port_for_scheme():
    assert has_non_standard_port_for_scheme("https://example.com:8443/") == 1
    assert has_non_standard_port_for_scheme("https://example.com:443/") == 0
    assert has_non_standard_port_for_scheme("http://example.com/") == 0


def test_has_sensitive_words_in_path():
    assert has_sensitive_words_in_path("http://example.com/account/login") == 1
    assert has_sensitive_words_in_path("http://example.com/about") == 0


def test_hostname_length():
    assert hostname_length("http://www.example.com/") == len("www.example.com")


def test_url_length():
    u = "http://example.com/"
    assert url_length(u) == len(u)


def test_hostname_shannon_entropy_higher_for_random_string():
    natural = hostname_shannon_entropy("https://www.google.com/")
    random_like = hostname_shannon_entropy("https://x7q9zpk3mvbnw.com/")
    assert random_like > natural


def test_longest_word_length_in_hostname():
    assert longest_word_length_in_hostname("http://verify-account-security-update.com") == len("security")


def test_extract_features_returns_all_names_in_order():
    feats = extract_features("https://www.example.com/path?x=1")
    assert list(feats.keys()) == FEATURE_NAMES
    assert all(isinstance(v, (int, float)) for v in feats.values())


@pytest.mark.parametrize("url", [
    "example.com",  # no scheme
    "http://",  # degenerate
    "",  # empty
])
def test_extract_features_does_not_crash_on_edge_cases(url):
    feats = extract_features(url)
    assert len(feats) == len(FEATURE_NAMES)

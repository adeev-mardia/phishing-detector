"""Lexical / URL feature extraction for phishing URL detection.

Every function here is a pure function of a URL string (no network calls,
no external state) so each one is independently unit-testable and the whole
module runs offline. The feature set follows the well-documented lexical
features used across published phishing-detection literature (e.g. the
UCI "Phishing Websites" dataset feature descriptions, and papers such as
Mohammad et al. 2014 "Intelligent Rule-based Phishing Websites
Classification", and Sahingoz et al. 2019).

`extract_features(url)` returns an ordered dict of ~20 numeric features
suitable for feeding straight into a scikit-learn model. `FEATURE_NAMES`
gives the fixed column order used everywhere else in the package.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import OrderedDict
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Reference data used by several heuristics
# ---------------------------------------------------------------------------

# A representative (not exhaustive) list of well-known URL shortener domains.
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "shorte.st", "cutt.ly", "rebrand.ly", "tiny.cc", "rb.gy",
    "shorturl.at", "bl.ink", "v.gd", "lnkd.in", "soo.gd", "s.id", "t.ly",
}

# Suspicious / frequently-abused TLDs in phishing campaigns (per multiple
# industry abuse reports, e.g. Spamhaus / Interisle "World's Most Abused
# TLDs" studies). Not exhaustive, but genuinely the commonly cited ones.
SUSPICIOUS_TLDS = {
    "zip", "review", "country", "kim", "cricket", "science", "work",
    "party", "gq", "link", "xyz", "top", "club", "tk", "ml", "ga", "cf",
    "men", "loan", "download", "racing", "win", "bid", "stream", "accountant",
}

# A short list of high-value brand names frequently impersonated in
# phishing attacks. Used only for the "brand name outside the registrable
# domain" heuristic below (e.g. brand in subdomain/path but not the real
# domain), a well-documented phishing tactic.
COMMON_BRANDS = {
    "paypal", "apple", "microsoft", "amazon", "google", "facebook",
    "netflix", "bankofamerica", "wellsfargo", "chase", "citibank", "hsbc",
    "instagram", "linkedin", "dropbox", "adobe", "ebay", "walmart",
    "americanexpress", "irs", "coinbase", "binance", "outlook", "office365",
}

IP_HOST_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
HEX_IP_RE = re.compile(r"^0x[0-9a-fA-F]+$")


def _safe_urlparse(url: str):
    url = url.strip()
    if "://" not in url:
        url = "http://" + url
    return urlparse(url)


def _hostname(url: str) -> str:
    parsed = _safe_urlparse(url)
    return (parsed.hostname or "").lower()


def _registrable_parts(hostname: str):
    """Very small heuristic splitter: returns (subdomain, domain, tld).

    This is intentionally simple (no public-suffix-list dependency, to keep
    the package dependency-free and offline) — good enough for lexical
    feature purposes, not meant to be a full PSL-accurate parser.
    """
    if not hostname:
        return "", "", ""
    labels = hostname.split(".")
    if len(labels) == 1:
        return "", labels[0], ""
    if len(labels) == 2:
        return "", labels[0], labels[1]
    # naive: last label = tld, second-to-last = domain, rest = subdomain
    tld = labels[-1]
    domain = labels[-2]
    subdomain = ".".join(labels[:-2])
    return subdomain, domain, tld


def has_ip_address(url: str) -> int:
    """1 if the host is a raw IPv4/IPv6/hex/octal address instead of a domain."""
    hostname = _hostname(url)
    if not hostname:
        return 0
    host = hostname.strip("[]")
    if IP_HOST_RE.match(host):
        return 1
    if HEX_IP_RE.match(host):
        return 1
    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        pass
    return 0


def url_length(url: str) -> int:
    return len(url.strip())


def hostname_length(url: str) -> int:
    return len(_hostname(url))


def has_at_symbol(url: str) -> int:
    """'@' in a URL causes browsers to ignore everything before it — classic trick."""
    return 1 if "@" in url else 0


def num_dots(url: str) -> int:
    return _hostname(url).count(".")


def num_hyphens_in_domain(url: str) -> int:
    hostname = _hostname(url)
    return hostname.count("-")


def num_subdomains(url: str) -> int:
    sub, _domain, _tld = _registrable_parts(_hostname(url))
    if not sub:
        return 0
    return len([p for p in sub.split(".") if p])


def uses_https(url: str) -> int:
    return 1 if _safe_urlparse(url).scheme == "https" else 0


def https_token_in_domain(url: str) -> int:
    """'https' or 'http' spelled inside the hostname itself (not the scheme) —
    e.g. http://https-paypal-secure.com/ — a documented phishing trick."""
    hostname = _hostname(url)
    return 1 if ("https" in hostname or "http" in hostname) else 0


def num_digits_in_url(url: str) -> int:
    return sum(ch.isdigit() for ch in url)


def digit_ratio(url: str) -> float:
    if not url:
        return 0.0
    return sum(ch.isdigit() for ch in url) / len(url)


def is_shortened_url(url: str) -> int:
    hostname = _hostname(url)
    return 1 if hostname in URL_SHORTENERS else 0


def has_suspicious_tld(url: str) -> int:
    _sub, _domain, tld = _registrable_parts(_hostname(url))
    return 1 if tld in SUSPICIOUS_TLDS else 0


def path_depth(url: str) -> int:
    path = _safe_urlparse(url).path or ""
    return len([p for p in path.split("/") if p])


def num_query_params(url: str) -> int:
    query = _safe_urlparse(url).query or ""
    if not query:
        return 0
    return len([p for p in query.split("&") if p])


def has_double_slash_redirect(url: str) -> int:
    """'//' appearing again after the scheme, later in the path — used to
    smuggle a redirect target, e.g. http://legit.com//evil.com"""
    parsed = _safe_urlparse(url)
    rest = parsed.path + ("?" + parsed.query if parsed.query else "")
    return 1 if "//" in rest else 0

def num_special_chars(url: str) -> int:
    return len(re.findall(r"[!$%^&*()_+|~=`{}\[\]:;\"'<>,]", url))


def brand_name_mismatch(url: str) -> int:
    """1 if a well-known brand name appears in the subdomain or path, but the
    registrable domain itself is NOT that brand's real domain. Classic
    tactic: paypal.com.login-secure.info or secure-paypal.evil.net."""
    hostname = _hostname(url)
    sub, domain, _tld = _registrable_parts(hostname)
    path = (_safe_urlparse(url).path or "").lower()
    haystack_outside_domain = f"{sub} {path}".lower()
    domain_lower = domain.lower()
    for brand in COMMON_BRANDS:
        if brand in haystack_outside_domain and brand != domain_lower:
            return 1
        # brand mashed into the domain but not as the domain itself, e.g.
        # "paypal-secure" as domain label
        if brand in domain_lower and domain_lower != brand:
            return 1
    return 0


def has_port_in_url(url: str) -> int:
    parsed = _safe_urlparse(url)
    return 1 if parsed.port is not None else 0


def hostname_shannon_entropy(url: str) -> float:
    """Shannon entropy of the hostname string. Randomly generated / DGA-like
    phishing domains tend to have higher character entropy than natural
    words."""
    hostname = _hostname(url)
    if not hostname:
        return 0.0
    freq = {}
    for ch in hostname:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(hostname)
    entropy = -sum((count / length) * math.log2(count / length) for count in freq.values())
    return round(entropy, 4)


def longest_word_length_in_hostname(url: str) -> int:
    """Longest alphabetic run in the hostname — phishing hostnames often
    concatenate many words/tokens (e.g. verify-account-security-update)."""
    hostname = _hostname(url)
    words = re.findall(r"[a-zA-Z]+", hostname)
    return max((len(w) for w in words), default=0)


def has_non_standard_port_for_scheme(url: str) -> int:
    parsed = _safe_urlparse(url)
    port = parsed.port
    if port is None:
        return 0
    default = 443 if parsed.scheme == "https" else 80
    return 1 if port != default else 0


def has_sensitive_words_in_path(url: str) -> int:
    """Words commonly seen in phishing URL paths (login/verify/update/etc.)."""
    keywords = (
        "login", "signin", "verify", "update", "secure", "account",
        "confirm", "banking", "password", "suspend", "unlock", "webscr",
    )
    path_and_query = (_safe_urlparse(url).path + "?" + _safe_urlparse(url).query).lower()
    return 1 if any(kw in path_and_query for kw in keywords) else 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "has_ip_address",
    "has_at_symbol",
    "num_dots",
    "num_hyphens_in_domain",
    "num_subdomains",
    "uses_https",
    "https_token_in_domain",
    "num_digits_in_url",
    "digit_ratio",
    "is_shortened_url",
    "has_suspicious_tld",
    "path_depth",
    "num_query_params",
    "has_double_slash_redirect",
    "num_special_chars",
    "brand_name_mismatch",
    "has_port_in_url",
    "hostname_shannon_entropy",
    "longest_word_length_in_hostname",
    "has_non_standard_port_for_scheme",
    "has_sensitive_words_in_path",
]


def extract_features(url: str) -> "OrderedDict[str, float]":
    """Extract the full feature vector (as an ordered dict) for one URL."""
    values = OrderedDict()
    values["url_length"] = url_length(url)
    values["hostname_length"] = hostname_length(url)
    values["has_ip_address"] = has_ip_address(url)
    values["has_at_symbol"] = has_at_symbol(url)
    values["num_dots"] = num_dots(url)
    values["num_hyphens_in_domain"] = num_hyphens_in_domain(url)
    values["num_subdomains"] = num_subdomains(url)
    values["uses_https"] = uses_https(url)
    values["https_token_in_domain"] = https_token_in_domain(url)
    values["num_digits_in_url"] = num_digits_in_url(url)
    values["digit_ratio"] = digit_ratio(url)
    values["is_shortened_url"] = is_shortened_url(url)
    values["has_suspicious_tld"] = has_suspicious_tld(url)
    values["path_depth"] = path_depth(url)
    values["num_query_params"] = num_query_params(url)
    values["has_double_slash_redirect"] = has_double_slash_redirect(url)
    values["num_special_chars"] = num_special_chars(url)
    values["brand_name_mismatch"] = brand_name_mismatch(url)
    values["has_port_in_url"] = has_port_in_url(url)
    values["hostname_shannon_entropy"] = hostname_shannon_entropy(url)
    values["longest_word_length_in_hostname"] = longest_word_length_in_hostname(url)
    values["has_non_standard_port_for_scheme"] = has_non_standard_port_for_scheme(url)
    values["has_sensitive_words_in_path"] = has_sensitive_words_in_path(url)
    assert list(values.keys()) == FEATURE_NAMES
    return values

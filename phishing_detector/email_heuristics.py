"""Rule-based phishing EMAIL analyzer.

This is deliberately separate from the URL model: it parses a raw .eml
(RFC 822) message, or plain text with a couple of extra fields, and scores
it against a genuine set of phishing heuristics used in real-world email
security tooling:

  - Urgency / pressure language ("act now", "account suspended", etc.)
  - Mismatched display text vs actual href in links
    (e.g. <a href="http://evil.tk">https://paypal.com</a>)
  - Sender display-name / domain mismatch (claims to be a brand, but the
    From: address domain doesn't match that brand)
  - Generic greeting ("Dear Customer") combined with a request for
    credentials or payment information
  - Excessive number of links
  - Links that use the URL-level phishing signals from features.py
    (IP host, suspicious TLD, brand mismatch, etc.)
  - Spoofed-looking Reply-To that differs from From

Returns a genuine weighted score (0-100) and a verdict, plus the list of
matched signals — not a stub.
"""

from __future__ import annotations

import email
import re
from dataclasses import dataclass, field
from email.message import Message
from typing import List, Optional, Tuple

from .features import brand_name_mismatch, has_ip_address, has_suspicious_tld

URGENCY_PHRASES = [
    "act now", "immediate action", "urgent", "verify your account",
    "account suspended", "account has been limited", "unusual activity",
    "confirm your identity", "your account will be closed",
    "click here immediately", "final notice", "action required",
    "within 24 hours", "your account has been locked", "security alert",
    "suspicious login", "expire", "limited time", "failure to comply",
]

CREDENTIAL_REQUEST_PHRASES = [
    "password", "social security", "ssn", "credit card", "card number",
    "cvv", "pin number", "login credentials", "update your billing",
    "confirm your password", "provide your account number", "wire transfer",
    "gift card", "bank details", "routing number",
]

GENERIC_GREETINGS = [
    "dear customer", "dear user", "dear valued customer", "dear member",
    "dear account holder", "dear sir/madam", "dear client",
]

BRAND_DOMAINS = {
    "paypal": "paypal.com",
    "apple": "apple.com",
    "microsoft": "microsoft.com",
    "amazon": "amazon.com",
    "google": "google.com",
    "facebook": "facebook.com",
    "netflix": "netflix.com",
    "bankofamerica": "bankofamerica.com",
    "wellsfargo": "wellsfargo.com",
    "chase": "chase.com",
    "irs": "irs.gov",
    "americanexpress": "americanexpress.com",
}

HREF_RE = re.compile(r'<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
URL_RE = re.compile(r'https?://[^\s"\'<>]+', re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class EmailSignal:
    name: str
    weight: int
    detail: str


@dataclass
class EmailAnalysis:
    score: int  # 0-100, higher = more likely phishing
    verdict: str  # "phishing", "suspicious", "legitimate"
    signals: List[EmailSignal] = field(default_factory=list)
    num_links: int = 0

    def as_dict(self):
        return {
            "score": self.score,
            "verdict": self.verdict,
            "num_links": self.num_links,
            "signals": [
                {"name": s.name, "weight": s.weight, "detail": s.detail}
                for s in self.signals
            ],
        }

    def pretty(self) -> str:
        lines = [f"Verdict: {self.verdict.upper()}  (score: {self.score}/100)"]
        if self.signals:
            lines.append("Matched signals:")
            for s in self.signals:
                lines.append(f"  [+{s.weight:>2}] {s.name}: {s.detail}")
        else:
            lines.append("No phishing signals detected.")
        return "\n".join(lines)


def _extract_sender_domain(from_header: str) -> Tuple[str, str]:
    """Return (display_name, domain) from a From: header value."""
    match = re.search(r"<([^>]+)>", from_header)
    addr = match.group(1) if match else from_header.strip()
    display = from_header.replace(f"<{addr}>", "").strip(' "')
    domain = addr.split("@")[-1].lower().strip() if "@" in addr else ""
    return display, domain


def _plain_text_from_message(msg: Message) -> str:
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype in ("text/plain", "text/html"):
                try:
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or "utf-8"
                    parts.append(payload.decode(charset, errors="replace"))
                except Exception:
                    continue
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload is not None:
                charset = msg.get_content_charset() or "utf-8"
                parts.append(payload.decode(charset, errors="replace"))
            else:
                parts.append(str(msg.get_payload()))
        except Exception:
            parts.append(str(msg.get_payload()))
    return "\n".join(parts)


def analyze_email(
    raw_email: Optional[str] = None,
    *,
    subject: str = "",
    sender: str = "",
    reply_to: str = "",
    body: str = "",
) -> EmailAnalysis:
    """Analyze an email for phishing signals.

    Either pass `raw_email` (a full .eml / RFC822 string) OR the individual
    fields (subject, sender, reply_to, body).
    """
    if raw_email:
        msg = email.message_from_string(raw_email)
        subject = msg.get("Subject", "") or subject
        sender = msg.get("From", "") or sender
        reply_to = msg.get("Reply-To", "") or reply_to
        body = _plain_text_from_message(msg) or body

    signals: List[EmailSignal] = []
    text_lower = (subject + "\n" + body).lower()
    body_no_tags = TAG_RE.sub(" ", body)

    # 1. Urgency / pressure language
    matched_urgency = [p for p in URGENCY_PHRASES if p in text_lower]
    if matched_urgency:
        signals.append(EmailSignal(
            "urgency_language", 15,
            f"{len(matched_urgency)} urgency phrase(s): {', '.join(matched_urgency[:4])}",
        ))

    # 2. Generic greeting + credential/payment request combo
    has_generic_greeting = any(g in text_lower for g in GENERIC_GREETINGS)
    matched_cred = [p for p in CREDENTIAL_REQUEST_PHRASES if p in text_lower]
    if has_generic_greeting and matched_cred:
        signals.append(EmailSignal(
            "generic_greeting_plus_credential_request", 20,
            f"generic greeting + request for: {', '.join(matched_cred[:4])}",
        ))
    elif matched_cred:
        signals.append(EmailSignal(
            "credential_or_payment_request", 10,
            f"requests: {', '.join(matched_cred[:4])}",
        ))

    # 3. Sender domain vs claimed brand mismatch
    display_name, sender_domain = _extract_sender_domain(sender)
    claimed_brand = None
    for brand, real_domain in BRAND_DOMAINS.items():
        if brand in display_name.lower() or brand in subject.lower() or brand in body_no_tags.lower()[:400]:
            claimed_brand = (brand, real_domain)
            break
    if claimed_brand:
        brand, real_domain = claimed_brand
        if sender_domain and not sender_domain.endswith(real_domain):
            signals.append(EmailSignal(
                "sender_domain_brand_mismatch", 25,
                f"message references '{brand}' but sender domain is '{sender_domain}' "
                f"(expected to end with '{real_domain}')",
            ))

    # 4. Reply-To differs from From domain (spoofing indicator)
    if reply_to:
        _rt_display, reply_domain = _extract_sender_domain(reply_to)
        if reply_domain and sender_domain and reply_domain != sender_domain:
            signals.append(EmailSignal(
                "reply_to_mismatch", 10,
                f"Reply-To domain '{reply_domain}' differs from From domain '{sender_domain}'",
            ))

    # 5. Links: mismatched display text vs href, and URL-level red flags
    hrefs_with_text = HREF_RE.findall(body)
    plain_urls = URL_RE.findall(body)
    all_link_targets = [h for h, _ in hrefs_with_text] + plain_urls
    num_links = len(set(all_link_targets))

    mismatched = 0
    for href, display_text in hrefs_with_text:
        shown_urls = URL_RE.findall(display_text)
        for shown in shown_urls:
            if shown.rstrip("/") != href.rstrip("/") and shown not in href:
                mismatched += 1
    if mismatched:
        signals.append(EmailSignal(
            "link_text_href_mismatch", 25,
            f"{mismatched} link(s) show one URL but point to a different href",
        ))

    risky_link_flags = 0
    for link in set(all_link_targets):
        try:
            if has_ip_address(link) or has_suspicious_tld(link) or brand_name_mismatch(link):
                risky_link_flags += 1
        except Exception:
            continue
    if risky_link_flags:
        signals.append(EmailSignal(
            "risky_link_url_signals", 20,
            f"{risky_link_flags} link(s) with IP-host/suspicious-TLD/brand-mismatch signals",
        ))

    if num_links >= 6:
        signals.append(EmailSignal(
            "excessive_links", 10, f"{num_links} distinct links in the message",
        ))

    # 6. Attachment-request-style language combined with urgency (common in
    # invoice/payment phishing)
    if "invoice" in text_lower and matched_urgency:
        signals.append(EmailSignal(
            "urgent_invoice_pattern", 10,
            "mentions an invoice together with urgency language",
        ))

    score = min(100, sum(s.weight for s in signals))
    if score >= 50:
        verdict = "phishing"
    elif score >= 20:
        verdict = "suspicious"
    else:
        verdict = "legitimate"

    return EmailAnalysis(score=score, verdict=verdict, signals=signals, num_links=num_links)

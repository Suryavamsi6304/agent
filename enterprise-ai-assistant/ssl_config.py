"""
SSL configuration for corporate proxy environments (Zscaler, etc.)

On corporate laptops, Zscaler acts as a TLS/SSL inspection proxy.
It intercepts HTTPS traffic and re-signs certificates with its own CA.
Python's bundled certifi CA store does not include the Zscaler CA,
which causes SSLError: certificate verify failed errors.

How to fix:
-----------
Option 1 (Recommended) — Point to your corporate CA bundle:
    Export REQUESTS_CA_BUNDLE=/path/to/zscaler-ca.crt

Option 2 — Disable SSL verification (not recommended for production):
    Export DISABLE_SSL_VERIFY=true

Option 3 — Use system certificates (macOS/Windows Zscaler installs the CA system-wide):
    Export USE_SYSTEM_CERTS=true

All options can be set in your .env file.
"""

import os
import ssl
import logging

logger = logging.getLogger(__name__)


def configure_ssl() -> None:
    """
    Apply SSL settings globally for corporate proxy environments.
    Call this once at application startup before any network calls.
    """
    disable_verify = os.environ.get("DISABLE_SSL_VERIFY", "").lower() in ("true", "1", "yes")
    use_system_certs = os.environ.get("USE_SYSTEM_CERTS", "").lower() in ("true", "1", "yes")
    ca_bundle = (
        os.environ.get("REQUESTS_CA_BUNDLE")
        or os.environ.get("SSL_CERT_FILE")
        or os.environ.get("CURL_CA_BUNDLE")
    )

    if disable_verify:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        # Propagate to all sub-libraries
        os.environ["PYTHONHTTPSVERIFY"] = "0"
        os.environ["CURL_CA_BUNDLE"] = ""
        os.environ["REQUESTS_CA_BUNDLE"] = ""
        logger.warning(
            "SSL verification is DISABLED. This is insecure — use only for local dev/testing."
        )
        return

    if ca_bundle and os.path.isfile(ca_bundle):
        # Propagate to requests, urllib3, httpx, and curl
        os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
        os.environ["SSL_CERT_FILE"] = ca_bundle
        os.environ["CURL_CA_BUNDLE"] = ca_bundle
        logger.info("SSL: using corporate CA bundle at %s", ca_bundle)
        return

    if use_system_certs:
        # Try to load the OS trust store (works on macOS, Windows with certifi-system)
        try:
            import certifi
            system_ca = certifi.where()
            os.environ.setdefault("REQUESTS_CA_BUNDLE", system_ca)
            os.environ.setdefault("SSL_CERT_FILE", system_ca)
            os.environ.setdefault("CURL_CA_BUNDLE", system_ca)
            logger.info("SSL: using certifi CA bundle at %s", system_ca)
        except ImportError:
            logger.warning("SSL: USE_SYSTEM_CERTS set but certifi not available.")


def get_ssl_verify():
    """
    Return the value to pass as `verify=` to httpx / requests.

    Returns:
        False              — if DISABLE_SSL_VERIFY=true
        str (path)         — if a CA bundle path is set
        True               — default (use bundled certifi)
    """
    if os.environ.get("DISABLE_SSL_VERIFY", "").lower() in ("true", "1", "yes"):
        return False

    ca_bundle = (
        os.environ.get("REQUESTS_CA_BUNDLE")
        or os.environ.get("SSL_CERT_FILE")
        or os.environ.get("CURL_CA_BUNDLE")
    )
    if ca_bundle and os.path.isfile(ca_bundle):
        return ca_bundle

    return True


def get_httpx_client():
    """
    Return an httpx.Client pre-configured for the corporate proxy/SSL environment.
    Use this when constructing the Groq SDK client.
    """
    import httpx

    verify = get_ssl_verify()

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    proxies = {"https://": proxy, "http://": proxy} if proxy else None

    return httpx.Client(verify=verify, proxies=proxies)


def get_async_httpx_client():
    """
    Return an httpx.AsyncClient pre-configured for the corporate proxy/SSL environment.
    """
    import httpx

    verify = get_ssl_verify()

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    proxies = {"https://": proxy, "http://": proxy} if proxy else None

    return httpx.AsyncClient(verify=verify, proxies=proxies)

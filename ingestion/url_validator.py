"""SSRF defense and URL validation for external content ingestion."""

import ipaddress
import socket
from urllib.parse import urlparse
from typing import Tuple
from utils.exceptions import SSRFSecurityError
from utils.logging_config import logger

# Blocked hostnames specifically known for cloud metadata or local resolution
BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "instance-data",
    "169.254.169.254",
}

def is_ip_forbidden(ip_str: str) -> bool:
    """Checks whether an IP address is private, loopback, link-local, multicast, or reserved."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        return True

def validate_target_url(
    url: str,
    allowed_schemes: Tuple[str, ...] = ("http", "https"),
    allowed_ports: Tuple[int, ...] = (80, 443, 8080, 8443),
    max_length: int = 2048,
) -> str:
    """
    Validates a URL against SSRF attacks before any network request is issued.
    
    Checks:
    - URL length
    - Valid scheme (http/https only)
    - Valid port
    - Non-empty hostname
    - DNS resolution to public, non-private, non-loopback IP addresses
    """
    if not url or not isinstance(url, str):
        raise SSRFSecurityError("URL cannot be empty or non-string.")

    clean_url = url.strip()
    if len(clean_url) > max_length:
        raise SSRFSecurityError(f"URL exceeds maximum allowed length of {max_length} characters.")

    try:
        parsed = urlparse(clean_url)
    except Exception as e:
        raise SSRFSecurityError(f"Malformed URL: {e}")

    scheme = parsed.scheme.lower()
    if scheme not in allowed_schemes:
        raise SSRFSecurityError(f"Forbidden URL scheme '{scheme}'. Allowed schemes: {allowed_schemes}")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFSecurityError("URL is missing a valid hostname.")

    hostname_lower = hostname.lower()
    if hostname_lower in BLOCKED_HOSTNAMES:
        raise SSRFSecurityError(f"Access to blocked hostname '{hostname}' is forbidden.")

    # Validate port
    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80

    if port not in allowed_ports:
        raise SSRFSecurityError(f"Forbidden port '{port}'. Allowed ports: {allowed_ports}")

    # Check if hostname directly represents an IP
    try:
        ipaddress.ip_address(hostname)
        if is_ip_forbidden(hostname):
            raise SSRFSecurityError(f"Direct access to private or restricted IP '{hostname}' is forbidden.")
    except ValueError:
        # Hostname is a domain name, resolve via DNS
        try:
            addr_info = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
            if not addr_info:
                raise SSRFSecurityError(f"Could not resolve hostname '{hostname}'.")

            for family, socktype, proto, canonname, sockaddr in addr_info:
                resolved_ip = sockaddr[0]
                if is_ip_forbidden(resolved_ip):
                    logger.warning(f"SSRF violation: Hostname '{hostname}' resolved to forbidden IP '{resolved_ip}'")
                    raise SSRFSecurityError(f"Hostname '{hostname}' resolves to a forbidden or private network IP.")
        except socket.gaierror as e:
            raise SSRFSecurityError(f"DNS resolution failure for '{hostname}': {e}")

    return clean_url


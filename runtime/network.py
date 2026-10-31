"""Use explicit proxy variables or the user's Windows Internet Settings."""
import os
import urllib.request


def configure_proxy():
    if os.name != "nt":
        return
    explicit = {key.lower() for key in os.environ}
    system = urllib.request.getproxies_registry()
    for scheme in ("http", "https"):
        key = scheme + "_proxy"
        if key not in explicit and system.get(scheme):
            os.environ[key.upper()] = system[scheme]
    urllib.request.install_opener(urllib.request.build_opener())

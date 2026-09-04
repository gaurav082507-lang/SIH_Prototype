# pfz_tool.py

import requests


PFZ_API_URL = "https://orca-backend-fz.onrender.com/api/pfz"


def get_pfz_data(latitude: float, longitude: float):
    """
    Fetch PFZ (Potential Fishing Zone) data from the
    ORCA PFZ backend.

    No LLM is used.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude
    }

    # timeout intentionally removed (per request) — this call will now
    # wait indefinitely for a response instead of failing after 30s.
    # NOTE: the PFZ backend drives a headless-Chrome/Puppeteer scrape of
    # INCOIS on every request and runs on a cold-starting Render free
    # tier, so it can legitimately take a long time — but with no
    # timeout at all, a genuinely stuck upstream request will hang this
    # node (and the whole graph run behind it) forever, with nothing to
    # catch or report. If that ever happens in practice, set an explicit
    # generous timeout (e.g. timeout=120) rather than leaving it
    # unbounded.
    response = requests.get(
        PFZ_API_URL,
        params=params,
        timeout=None
    )

    response.raise_for_status()

    data = response.json()

    return data

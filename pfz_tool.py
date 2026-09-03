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

    response = requests.get(
        PFZ_API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    return data
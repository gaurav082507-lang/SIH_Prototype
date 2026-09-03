
import requests
from langchain_core.tools import tool


ECOSYSTEM_API_URL = (
    "https://orca-backend-ecosystem.onrender.com/api/ecosystem"
)


@tool
def get_ecosystem_data(
    latitude: float,
    longitude: float,
    date: str
) -> dict:
    """
    Fetch marine ecosystem data for a latitude,
    longitude and date.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "date": date
    }

    response = requests.get(
        ECOSYSTEM_API_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    return response.json()


def get_ecosystem_data_for_grid(
    locations: list,
    date: str
) -> list:
    """
    Fetch ecosystem data for all planner grid points.

    locations format:

    [
        {
            "id": "P1",
            "latitude": 19.1661,
            "longitude": 72.8777,
            "bearing_deg": 0,
            "distance_km": 10
        },
        ...
    ]
    """

    results = []

    for location in locations:

        latitude = location["latitude"]
        longitude = location["longitude"]

        data = get_ecosystem_data.invoke({
            "latitude": latitude,
            "longitude": longitude,
            "date": date
        })

        # Keep planner grid-point information
        data["id"] = location.get("id")
        data["bearing_deg"] = location.get("bearing_deg")
        data["distance_km"] = location.get("distance_km")

        results.append(data)

    return results
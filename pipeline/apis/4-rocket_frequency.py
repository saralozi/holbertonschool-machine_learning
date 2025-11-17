#!/usr/bin/env python3
"""A script that displays the number of launches per rocket."""

import requests


def rocket_frequency():
    """A function that displays the number of launches per rocket"""

    base = "https://api.spacexdata.com"

    launches = requests.get(f"{base}/v4/launches", timeout=10).json()

    counts = {}
    for launch in launches:
        rocket_id = launch["rocket"]
        counts[rocket_id] = counts.get(rocket_id, 0) + 1

    rockets = {}
    for rocket_id in counts:
        rockets[rocket_id] = requests.get(
            f"{base}/v4/rockets/{rocket_id}",
            timeout=10
        ).json()["name"]

    results = [
        (rockets[rocket_id], count)
        for rocket_id, count in counts.items()
    ]

    results.sort(key=lambda x: (-x[1], x[0]))
    return results


if __name__ == "__main__":
    for name, count in rocket_frequency():
        print(f"{name}: {count}")

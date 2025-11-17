#!/usr/bin/env python3
"""A script that displays the number of launches per rocket."""

import requests


def rocket_frequency():
    """A function that displays the number of launches per rocket"""

    url = "https://api.spacexdata.com"

    launches = requests.get(f"{url}/v4/launches").json()

    counts = {}
    for launch in launches:
        rocket_id = launch["rocket"]

        if rocket_id in counts:
            counts[rocket_id] += 1
        else:
            counts[rocket_id] = 1

    names = {}
    for rocket_id in counts:
        data = requests.get(f"{url}/v4/rockets/{rocket_id}").json()
        names[rocket_id] = data["name"]

    results = []
    for rocket_id, count in counts.items():
        rocket_name = names[rocket_id]
        results.append((rocket_name, count))

        results.sort(key=lambda item: (-item[1], item[0]))
    return results


if __name__ == "__main__":
    for name, count in rocket_frequency():
        print(f"{name}: {count}")

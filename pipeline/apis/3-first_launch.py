#!/usr/bin/env python3
"""A script that  displays the first launch with some information"""


import requests


def first_launch():
    """A function that displays the first launch with some information"""

    url = "https://api.spacexdata.com"

    launches = requests.get(f"{url}/v5/launches", timeout=10).json()
    first = min(launches, key=lambda launch: launch["date_unix"])

    launch_name = first["name"]
    local_date = first["date_local"]
    rocket_id = first["rocket"]
    launchpad_id = first["launchpad"]

    rockets = requests.get(f"{url}/v4/rockets/{rocket_id}").json()
    rocket_name = rockets['name']

    launchapds = requests.get(f"{url}/v4/launchpads/{launchpad_id}").json()
    launchpad_name = launchapds["name"]
    launchpad_locality = launchapds["locality"]

    return (
        f"{launch_name} ({local_date}) {rocket_name} - "
        f"{launchpad_name} ({launchpad_locality})"
    )


if __name__ == "__main__":
    print(first_launch())

#!/usr/bin/env python3
"""A script that  displays the upcoming launch with some information"""


import requests


def upcoming_launch():
    """A function that displays the upcoming launch with some information"""

    url = "https://api.spacexdata.com"

    launches = requests.get(f"{url}/v5/launches", timeout=10).json()

    upcoming = [launch for launch in launches if launch["upcoming"]]
    upcoming.sort(key=lambda launch: launch["date_unix"])

    name = upcoming[0]["name"]
    date = upcoming[0]["date_local"]
    rocket_id = upcoming[0]["rocket"]
    launchpad_id = upcoming[0]["launchpad"]

    rockets = requests.get(f"{url}/v4/rockets/{rocket_id}").json()
    rocket_name = rockets['name']

    launchapds = requests.get(f"{url}/v4/launchpads/{launchpad_id}").json()
    launchpad_name = launchapds["name"]
    launchpad_locality = launchapds["locality"]

    return (
        f"{name} ({date}) {rocket_name} - "
        f"{launchpad_name} ({launchpad_locality})"
    )


if __name__ == "__main__":
    print(upcoming_launch())

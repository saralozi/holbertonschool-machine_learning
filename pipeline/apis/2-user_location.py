#!/usr/bin/env python3
"""A script that prints the location of a specific user"""

import sys
import requests
import time


def user_location(url):
    """A function that prints the location of a specific user"""

    response = requests.get(url)

    if response.status_code == 404:
        return "Not found"

    if response.status_code == 403:
        reset_ts = int(response.headers.get("X-RateLimit-Reset", 0))
        now = int(time.time())
        x = max((reset_ts - now) // 60, 0)

        return (f"Reset in {x} min")

    if response.status_code == 200:
        location = response.json().get('location')
        return location

    return None


if __name__ == "__main__":
    url = sys.argv[1]
    result = user_location(url)
    if result:
        print(result)

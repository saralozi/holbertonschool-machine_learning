#!/usr/bin/env python3
"""A script that returns the list of ships that
can hold a given number of passengers"""


import requests
import time


def availableShips(passengerCount):
    """A function that returns the list of ships than
    can hold a given number of passengers"""

    url = "https://swapi-api.hbtn.io/api/starships/"
    ships = []

    while url:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            print("Error fetching data")
            return []

        else:
            data = response.json()

            for ship in data['results']:

                passengers = ship.get('passengers', '0')

                passengers = passengers.replace(',', '')

                if not passengers.isdigit():
                    continue

                if int(passengers) >= passengerCount:
                    ships.append(ship['name'])

            time.sleep(1)
            url = data.get('next')

    return ships

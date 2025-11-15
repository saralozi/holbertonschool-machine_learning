#!/usr/bin/env python3
"""A script that returns the list of names of the
home planets of all sentient species"""


import requests
import time


def sentientPlanets():
    """A function that returns the list of names of the
    home planets of all sentient species"""

    url = "https://swapi-api.hbtn.io/api/species/"
    home_planets = []

    while url:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            print("Error parsing data")
            return []
        else:
            data = response.json()

            for species in data.get('results', []):
                classification = species.get('classification', '').lower()
                designation = species.get('designation', '').lower()

                if (
                    'sentient' not in classification
                    and 'sentient' not in designation
                ):
                    continue

                homeworld_url = species.get('homeworld')

                if not homeworld_url:
                    continue

                planet_response = requests.get(homeworld_url, timeout=10)

                if planet_response.status_code != 200:
                    continue

                planet_data = planet_response.json()
                planet_name = planet_data.get('name')
                planet_url = planet_data.get('url')

                if planet_url:
                    planet_id = int(planet_url.rstrip('/').split('/')[-1])
                else:
                    continue

                if (
                    planet_name and planet_name not in
                    [p[1] for p in home_planets]
                ):
                    home_planets.append((planet_id, planet_name))

            time.sleep(1)
            url = data.get('next')

    home_planets.sort(key=lambda x: x[0])

    return [name for _, name in home_planets]

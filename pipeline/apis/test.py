import requests
import json

url = "https://swapi-api.hbtn.io/api/species/"

response = requests.get(url)

data = response.json()
first_species = data['results'][0]
print(json.dumps(first_species, indent=2))
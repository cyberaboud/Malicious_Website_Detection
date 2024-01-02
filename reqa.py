import requests
import json

# URL of the API endpoint
url = 'http://127.0.0.1:5000/model'

# Data to be sent in the POST request
data = {
    "url": "https://example.com"  # Replace this with the URL you want to classify
}

# Convert data to JSON format
json_data = json.dumps(data)
print(json_data)

# Set the headers
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}

# Make the POST request
response = requests.post(url, data=json_data, headers=headers)

# Check the response
if response.status_code == 200:
    result = response.json()
    print(result)  # Output the prediction result
else:
    print("Request failed with status code:", response.status_code)
    print("Response content:", response.text)

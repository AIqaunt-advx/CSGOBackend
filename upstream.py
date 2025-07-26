import requests
import json

# Example data matching the TrendDetailsDataItem structure
# You would replace this with your actual data
sample_data = [
    {
        "timestamp": 1678886400,
        "price": 100.0,
        "onSaleQuantity": 50,
        "seekPrice": 95.0,
        "seekQuantity": 10,
        "transactionAmount": 1000.0,
        "transcationNum": 10,
        "surviveNum": 5
    },
    {
        "timestamp": 1678886460,
        "price": 102.0,
        "onSaleQuantity": 45,
        "seekPrice": 97.0,
        "seekQuantity": 12,
        "transactionAmount": 1200.0,
        "transcationNum": 12,
        "surviveNum": 6
    }
]

# API endpoint
url = "http://localhost:8000/predict"

# Prepare the request payload
payload = {"data": sample_data}

# Send the POST request
try:
    response = requests.post(url, json=payload)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

    # Parse the JSON response
    result = response.json()
    print("Predictions:", result.get("predictions"))
    print("MSE:", result.get("mse"))

except requests.exceptions.RequestException as e:
    print(f"Error making request: {e}")
except json.JSONDecodeError:
    print("Error decoding JSON response.")
    print("Response content:", response.text)

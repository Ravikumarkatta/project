import requests
import json

BASE_URL = "http://localhost:8000"

def health_check():
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())

def single_prediction():
    data = {
        "text": "For God so loved the world",
        "context": "John 3:16"
    }
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data
    )
    print("Single Prediction:", json.dumps(response.json(), indent=2))

def batch_prediction():
    data = {
        "texts": [
            "In the beginning God created",
            "The Lord is my shepherd"
        ],
        "contexts": [
            "Genesis 1:1",
            "Psalm 23:1"
        ]
    }
    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json=data
    )
    print("Batch Prediction:", json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    health_check()
    single_prediction()
    batch_prediction()

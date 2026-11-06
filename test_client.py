import requests
import json

# The URL of the API
API_URL = "http://127.0.0.1:7860/predict"

IMAGE_PATH = "assets/01_pinhole_05.jpg"

def test_api():
    print(f"Sending request to the server with: {IMAGE_PATH}")
    
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": (IMAGE_PATH, f, "image/jpeg")}
        
        response = requests.post(API_URL, files=files)
    
    # If the server returns a successful (200) response, print the JSON output
    if response.status_code == 200:
        print("\n--- PREDICTION RESULTS ---")
  
        print(json.dumps(response.json(), indent=4, ensure_ascii=False))
    else:
        print(f"Error! Status Code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_api()
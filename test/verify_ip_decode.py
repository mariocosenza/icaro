import requests
from urllib.parse import quote

BASE_URL = "http://localhost:8000/api/v1/raspberry-ip"

def test_update_ip(address):
    print(f"Testing address: {address}")
    # We pass it as a query parameter
    url = f"{BASE_URL}?address={quote(address)}"
    try:
        response = requests.patch(url)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_update_ip("192.168.1.100")
    test_update_ip("http://raspberrypi.local:5000")

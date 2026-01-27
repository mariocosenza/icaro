# Icaro Buzzer Server

This is a lightweight FastAPI server designed to run on a Raspberry Pi to control a buzzer (or other GPIO device) in response to HTTP requests.

## Prerequisites

- Raspberry Pi (Zero, 3, 4, 5, etc.)
- Python 3.9+
- Active Buzzer connected to a GPIO pin (default GPIO 17)

This server runs a continuous buzzer alert that must be manually stopped via a POST request.

## Installation

1.  **Transfer files**: Copy the `server/` directory to your Raspberry Pi.
2.  **Install dependencies**:
    ```bash
    cd server
    pip install -r requirements.txt
    ```

## Usage

### Running Locally (Manual)

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Running as a Service (Auto-start)

1.  **Edit the Service File**: Open `icaro-buzzer.service` and adjust the paths and user if necessary.
    - `User=pi` (Change if your username is different)
    - `WorkingDirectory=/home/pi/icaro/server` (Set to your actual path)
    - `ExecStart=/usr/bin/python3 ...` (Ensure python path is correct)

2.  **Install Service**:
    ```bash
    sudo cp icaro-buzzer.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable icaro-buzzer.service
    sudo systemctl start icaro-buzzer.service
    ```

3.  **Check Status**:
    ```bash
    sudo systemctl status icaro-buzzer.service
    ```

## API Endpoints

-   `GET /health`: Checks status.
-   `POST /alert`: Triggers the buzzer to run CONTINUOUSLY.
    -   Body: `{"message": "optional text"}`
-   `POST /stop`: Stops the buzzer.

## Testing

You can run the unit tests (mocked GPIO) on any machine:

```bash
pytest server/tests/test_main.py
```

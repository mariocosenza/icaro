import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
BUZZER_PIN = int(os.getenv("BUZZER_PIN", 17))
BUZZ_DURATION = float(os.getenv("BUZZ_DURATION", 5.0))  # seconds

# Global buzzer instance
buzzer = None
has_gpio = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize GPIO
    global buzzer, has_gpio
    try:
        from gpiozero import Buzzer
        from gpiozero.pins.mock import MockFactory
        
        # Check if we are running on a Pi or need to mock
        try:
           buzzer = Buzzer(BUZZER_PIN)
           has_gpio = True
           logger.info(f"GPIO initialized on pin {BUZZER_PIN}")
        except Exception as e:
           logger.warning(f"Could not initialize GPIO (running on non-Pi device?): {e}")
           has_gpio = False

    except ImportError:
        logger.warning("gpiozero library not found. Running in simulation mode.")
        has_gpio = False
    
    yield
    
    # Shutdown: Cleanup if needed (gpiozero handles cleanup mostly)
    if buzzer:
        buzzer.close()

app = FastAPI(lifespan=lifespan)

class AlertRequest(BaseModel):
    message: str | None = None
    duration: float | None = None

async def play_buzzer_pattern(duration: float):
    """
    Buzzes for the specified duration.
    Can be enhanced to play patterns (beep-beep-beep).
    """
    logger.info(f"Triggering buzzer for {duration} seconds")
    
    if has_gpio and buzzer:
        end_time = asyncio.get_event_loop().time() + duration
        while asyncio.get_event_loop().time() < end_time:
            buzzer.on()
            await asyncio.sleep(15)
            buzzer.off()
            await asyncio.sleep(15)
    else:
        logger.info(f"[MOCK] Buzzer ON-OFF pattern for {duration} seconds")
        await asyncio.sleep(duration)
    
    logger.info("Buzzer sequence finished")

@app.post("/alert")
async def trigger_alert(request: AlertRequest, background_tasks: BackgroundTasks):
    """
    Receives an alert request and triggers the buzzer in the background.
    """
    duration = request.duration if request.duration is not None else BUZZ_DURATION
    background_tasks.add_task(play_buzzer_pattern, duration)
    return {"status": "success", "message": "Alert triggered", "duration": duration}

@app.get("/health")
async def health_check():
    return {"status": "online", "gpio_active": has_gpio}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

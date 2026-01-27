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

# Global buzzer instance
buzzer = None
has_gpio = False
buzzer_task = None
stop_event = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize GPIO
    global buzzer, has_gpio, stop_event
    
    # Initialize event here to ensure it uses the running event loop
    stop_event = asyncio.Event()

    try:
        from gpiozero import Buzzer
        
        # Check if we are running on a Pi
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
    
    # Shutdown: Stop any running buzzer task and cleanup
    stop_event.set()
    if buzzer_task:
        try:
            await buzzer_task
        except asyncio.CancelledError:
            pass
            
    if buzzer:
        buzzer.close()

app = FastAPI(lifespan=lifespan)

class AlertRequest(BaseModel):
    message: str | None = None

async def continuous_buzz():
    """
    Buzzes continuously until stop_event is set.
    """
    logger.info("Starting continuous buzzer pattern")
    try:
        while not stop_event.is_set():
            if has_gpio and buzzer:
                buzzer.on()
            else:
                logger.info("[MOCK] BEEP")
            
            # Beep ON duration
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=0.5)
                if stop_event.is_set(): break
            except asyncio.TimeoutError:
                pass

            if has_gpio and buzzer:
                buzzer.off()
            
            # Beep OFF duration
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass
                
    except asyncio.CancelledError:
        logger.info("Buzzer task cancelled")
    finally:
        # Ensure buzzer is off
        if has_gpio and buzzer:
            buzzer.off()
        logger.info("Buzzer stopped")

@app.post("/alert")
async def trigger_alert(request: AlertRequest):
    """
    Triggers the buzzer to run continuously until stopped.
    """
    global buzzer_task
    
    if buzzer_task and not buzzer_task.done():
        return {"status": "ignored", "message": "Buzzer already active"}
    
    stop_event.clear()
    buzzer_task = asyncio.create_task(continuous_buzz())
    return {"status": "success", "message": "Buzzer started"}

@app.post("/stop")
async def stop_alert():
    """
    Stops the buzzer.
    """
    global buzzer_task
    
    if not buzzer_task or buzzer_task.done():
         return {"status": "ignored", "message": "Buzzer is not active"}

    stop_event.set()
    await buzzer_task
    return {"status": "success", "message": "Buzzer stopped"}

@app.get("/health")
async def health_check():
    return {
        "status": "online", 
        "gpio_active": has_gpio,
        "buzzer_active": (buzzer_task is not None and not buzzer_task.done())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

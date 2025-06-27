from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict
import reward_engine
import logging

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI(title="Karma Surprise Box API", description="AI-driven reward microservice", version="1.0.0")

# Pydantic Model for Input
class DailyMetrics(BaseModel):
    login_streak: int = Field(..., ge=0, description="Number of consecutive days the user has logged in")
    posts_created: int = Field(..., ge=0, description="Number of posts created today")
    comments_written: int = Field(..., ge=0, description="Number of comments written today")
    upvotes_received: int = Field(..., ge=0, description="Number of upvotes received today")
    quizzes_completed: int = Field(..., ge=0, description="Number of quizzes completed today")
    buddies_messaged: int = Field(..., ge=0, description="Number of buddies messaged today")
    karma_spent: int = Field(..., ge=0, description="Amount of karma spent today")
    karma_earned_today: int = Field(..., ge=0, description="Amount of karma earned today")

class SurpriseBoxRequest(BaseModel):
    user_id: str
    date: str
    daily_metrics: DailyMetrics

# Health Check
@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "ok"}

# Version Info
@app.get("/version")
async def version_info():
    logger.info("Version info requested")
    return {"model_version": reward_engine.reward_model_handler.model_version, "feature_names_version": reward_engine.reward_model_handler.feature_names_version}

# Check Surprise Box
@app.post("/check-surprise-box")
async def check_surprise_box(request: SurpriseBoxRequest):
    logger.info(f"Received request for user_id: {request.user_id}, date: {request.date}")
    try:
        daily_metrics_dict = request.daily_metrics.dict()
        response = reward_engine.determine_reward_details(request.user_id, request.date, daily_metrics_dict)
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=400 if isinstance(e, ValueError) else 500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Karma Surprise Box service on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
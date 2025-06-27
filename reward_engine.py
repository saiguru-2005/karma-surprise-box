import random
import json
import joblib
import pandas as pd
import logging
from typing import Dict, Any
import sqlite3

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Configuration
MODEL_PATH = "classifier_new.pkl"
FEATURES_PATH = "feature_names_new.json"
CONFIG_PATH = "config.json"

CONFIG_KARMA_MIN = 5
CONFIG_KARMA_MAX = 30

# Singleton for model handling
class RewardModelHandler:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.load_error = None
        self.model_version = MODEL_PATH
        self.feature_names_version = FEATURES_PATH
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            self.model = joblib.load(MODEL_PATH)
            logger.info(f"Loading feature names from {FEATURES_PATH}")
            with open(FEATURES_PATH, 'r') as f:
                self.feature_names = json.load(f)
            logger.info(f"AI Model ('{MODEL_PATH}') and {len(self.feature_names)} feature names loaded successfully.")
            logger.debug(f"Feature names: {self.feature_names}")
        except Exception as e:
            self.load_error = str(e)
            logger.error(f"CRITICAL ERROR: Could not load AI model or feature names file.")
            logger.error(f"Model Path: {MODEL_PATH}")
            logger.error(f"Features Path: {FEATURES_PATH}")
            logger.error(f"Error: {e}")
            logger.error(f"The reward engine will operate in a degraded (rules-only or no-AI) mode.")

reward_model_handler = RewardModelHandler()

# Load configuration
try:
    logger.info(f"Loading configuration from {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    REWARD_RULES = sorted(config.get("rules", []), key=lambda x: x["reward_karma"], reverse=True)  # Sort by reward_karma
    ENABLE_AI_REWARDS = config.get("enable_ai_rewards", False)
    MYSTERY_REWARD = config.get("mystery_reward", {})
    logger.info("Configuration loaded successfully.")
    logger.debug(f"Reward rules: {REWARD_RULES}")
    logger.debug(f"Enable AI rewards: {ENABLE_AI_REWARDS}")
    logger.debug(f"Mystery reward: {MYSTERY_REWARD}")
except Exception as e:
    logger.error(f"Failed to load configuration from {CONFIG_PATH}: {str(e)}")
    raise

# Database setup
def init_db():
    conn = sqlite3.connect('reward_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reward_history
                 (user_id TEXT, date TEXT, reward_delivered INTEGER)''')
    conn.commit()
    conn.close()

init_db()  # Initialize database on module load

def check_reward_history(user_id: str, date_str: str) -> bool:
    conn = sqlite3.connect('reward_history.db')
    c = conn.cursor()
    c.execute("SELECT reward_delivered FROM reward_history WHERE user_id = ? AND date = ?", (user_id, date_str))
    result = c.fetchone()
    conn.close()
    return result is not None and result[0] == 1

def update_reward_history(user_id: str, date_str: str, delivered: bool):
    conn = sqlite3.connect('reward_history.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO reward_history (user_id, date, reward_delivered) VALUES (?, ?, ?)",
              (user_id, date_str, 1 if delivered else 0))
    conn.commit()
    conn.close()

def determine_reward_details(user_id: str, date_str: str, daily_metrics_input: Dict[str, Any]) -> Dict[str, Any]:
    logger.info(f"Determining reward for user_id: {user_id}, date: {date_str}")
    logger.debug(f"Input metrics: {daily_metrics_input}")

    # Check if user has already received a reward today
    if check_reward_history(user_id, date_str):
        logger.info(f"User {user_id} already received a reward on {date_str}, skipping.")
        return _build_response(user_id, False, 0, None, None, None, "missed")

    # Validate required fields
    required_fields = {"login_streak", "quizzes_completed", "posts_created", "comments_written", "upvotes_received", "buddies_messaged", "karma_spent", "karma_earned_today"}
    if not all(field in daily_metrics_input for field in required_fields):
        missing = required_fields - set(daily_metrics_input.keys())
        logger.error(f"Missing required fields: {missing}")
        return _build_response(user_id, False, 0, "Invalid input: missing fields", None, None, "missed")

    # Normalize metrics to integers
    daily_metrics = {k: int(v) if isinstance(v, (str, int)) else 0 for k, v in daily_metrics_input.items()}
    logger.debug(f"Normalized input metrics: {daily_metrics}")

    # Deterministic Seeding
    seed_value_str = f"{user_id}-{date_str}"
    current_seed = sum(ord(c) * (i+1) for i, c in enumerate(seed_value_str))
    random.seed(current_seed)
    logger.debug(f"Seed value: {current_seed}")

    surprise_unlocked = False
    reward_karma = 0
    reason = ""
    rarity = None
    box_type = None
    status = "missed"

    # Evaluate all rules and select the best match based on reward_karma
    best_reward = 0
    for rule in REWARD_RULES:
        try:
            condition = rule["condition"].strip()
            logger.debug(f"Evaluating rule: {condition}")
            condition_parts = [part.strip() for part in condition.split(" and ")]
            condition_met = True
            for part in condition_parts:
                if ">=" in part:
                    key, value = part.split(">=")
                    key = key.strip()
                    value = int(value.strip())
                    actual_value = int(daily_metrics.get(key, 0))
                    logger.debug(f"Evaluating {key} >= {value}: actual value = {actual_value}")
                    if actual_value < value:
                        condition_met = False
                        break
            if condition_met and rule["reward_karma"] > best_reward:
                surprise_unlocked = True
                best_reward = rule["reward_karma"]
                reward_karma = rule["reward_karma"]
                reason = rule["reason"]
                rarity = rule["rarity"]
                box_type = rule["box_type"]
                status = "delivered"
                logger.info(f"Rule matched for user '{user_id}': rarity={rarity}, box_type={box_type}")
        except Exception as e:
            logger.error(f"Error evaluating rule {rule['condition']}: {str(e)}")
            continue

    # Mystery Reward as a fallback with 10% chance
    if not surprise_unlocked and daily_metrics.get("login_streak", 0) >= 1 and random.random() < MYSTERY_REWARD.get("probability", 0.10):
        surprise_unlocked = True
        reward_karma = MYSTERY_REWARD.get("reward_karma", 5)
        reason = MYSTERY_REWARD.get("reason", "A lucky bonus for your activity!")
        rarity = MYSTERY_REWARD.get("rarity", "common")
        box_type = MYSTERY_REWARD.get("box_type", "mystery_luck")
        status = "delivered"
        logger.info(f"Mystery reward for user '{user_id}': rarity={rarity}, box_type={box_type}")

    # Mark user as rewarded if a reward was granted
    if surprise_unlocked:
        update_reward_history(user_id, date_str, True)

    return _build_response(user_id, surprise_unlocked, reward_karma, reason, rarity, box_type, status)

def _build_response(user_id, unlocked, karma, reason_str, rarity_val, box_type_val, status_val):
    """Helper function to build the response dictionary."""
    response = {
        "user_id": user_id,
        "surprise_unlocked": unlocked,
        "reward_karma": karma if unlocked else 0,
        "reason": reason_str if unlocked else None,
        "rarity": rarity_val if unlocked else None,
        "box_type": box_type_val if unlocked else None,
        "status": status_val
    }
    logger.debug(f"Response: {response}")
    return response
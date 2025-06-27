import random
import json
import datetime
from asteval import Interpreter

# Load configuration from config.json
with open("config.json", "r") as f:
    config = json.load(f)

# Configuration parameters
NUM_USERS = config.get("num_users", 500)
DAYS_PER_USER = config.get("days_per_user", 100)
OUTPUT_FILENAME = config.get("output_filename", "training_data.json")
START_DATE = datetime.date.fromisoformat(config.get("start_date", "2024-01-01"))
USER_TYPE_PROBS = config.get("user_types", {"casual": 0.5, "engaged": 0.3, "power": 0.2})
REWARD_RULES = config.get("reward_rules", [])

# Metric generation parameters per user type
METRIC_PARAMS = {
    "login_streak": {
        "continuation_prob": {"casual": 0.8, "engaged": 0.9, "power": 0.95},
        "start_prob": {"casual": 0.3, "engaged": 0.5, "power": 0.7},
        "max": 150
    },
    "posts_created": {
        "lambda": {"casual": 1.5, "engaged": 0.9, "power": 0.3},
        "max": 10
    },
    "comments_written": {
        "lambda": {"casual": 1.0, "engaged": 0.6, "power": 0.2},
        "max": 25
    },
    "upvotes_received": {
        "lambda": {"casual": 0.1, "engaged": 0.08, "power": 0.05},
        "max": 100
    },
    "quizzes_completed": {
        "lambda": {"casual": 1.2, "engaged": 0.8, "power": 0.4},
        "max": 5
    },
    "buddies_messaged": {
        "lambda": {"casual": 0.6, "engaged": 0.4, "power": 0.2},
        "max": 30
    },
    "karma_spent": {
        "lambda": {"casual": 0.1, "engaged": 0.05, "power": 0.02},
        "max": 200
    },
    "karma_earned_today": {
        "lambda": {"casual": 0.09, "engaged": 0.05, "power": 0.02},
        "max": 150
    }
}

def generate_daily_activity_metrics(user_type, previous_day_metrics, current_date):
    """Generate daily metrics based on user type and previous day's data."""
    metrics = {}

    # Login streak depends on previous day
    if previous_day_metrics and previous_day_metrics.get("login_streak", 0) > 0:
        if random.random() < METRIC_PARAMS["login_streak"]["continuation_prob"][user_type]:
            metrics["login_streak"] = min(previous_day_metrics["login_streak"] + 1, METRIC_PARAMS["login_streak"]["max"])
        else:
            metrics["login_streak"] = 0
    else:
        if random.random() < METRIC_PARAMS["login_streak"]["start_prob"][user_type]:
            metrics["login_streak"] = 1
        else:
            metrics["login_streak"] = 0

    # Generate other metrics with exponential distribution
    for metric, params in METRIC_PARAMS.items():
        if metric != "login_streak":
            p = params["lambda"][user_type]
            metrics[metric] = min(int(random.expovariate(p)), params["max"])

    # Dependency: upvotes_received requires posts or comments
    if metrics.get("posts_created", 0) > 0 or metrics.get("comments_written", 0) > 2:
        p = METRIC_PARAMS["upvotes_received"]["lambda"][user_type]
        metrics["upvotes_received"] = min(int(random.expovariate(p)), METRIC_PARAMS["upvotes_received"]["max"])
    else:
        metrics["upvotes_received"] = 0

    return metrics

def determine_reward_from_metrics(daily_metrics, reward_rules):
    """Determine if a reward is granted based on metrics and configurable rules."""
    aeval = Interpreter()
    aeval.symtable.update(daily_metrics)
    for rule in reward_rules:
        try:
            if aeval(rule["condition"]) and random.random() < rule["probability"]:
                karma_reward = random.randint(rule["karma_min"], rule["karma_max"])
                return (1, karma_reward, rule["box_type"], rule["rarity"], rule["reason"])
        except Exception as e:
            print(f"Error evaluating rule {rule['condition']}: {e}")
            continue
    return (0, 0, None, None, "No qualifying pattern")

# Assign user types
user_types = random.choices(list(USER_TYPE_PROBS.keys()), weights=list(USER_TYPE_PROBS.values()), k=NUM_USERS)
users = [{"id": f"sim_user_{i:04d}", "type": user_type} for i, user_type in enumerate(user_types)]

# Generate dataset
all_samples = []
for user in users:
    previous_day_data = None
    for day_num in range(DAYS_PER_USER):
        current_date = START_DATE + datetime.timedelta(days=day_num)
        metrics = generate_daily_activity_metrics(user["type"], previous_day_data, current_date)
        label, reward_score, box_type, rarity, reason = determine_reward_from_metrics(metrics, REWARD_RULES)
        sample = {
            "user_id": user["id"],
            "date": current_date.isoformat(),
            "features": metrics,
            "surprise_unlocked": label == 1,
            "reward_karma": reward_score if label == 1 else 0,
            "reason": reason,
            "rarity": rarity if label == 1 else None,
            "box_type": box_type if label == 1 else None,
            "status": "delivered" if label == 1 else "missed"
        }
        all_samples.append(sample)
        previous_day_data = metrics

# Validation function to catch issues
def validate_dataset(samples):
    issues = []
    for sample in samples:
        metrics = sample["features"]
        if metrics["upvotes_received"] > 0 and metrics["posts_created"] == 0 and metrics["comments_written"] <= 2:
            issues.append(f"User {sample['user_id']} on {sample['date']}: Upvotes without posts or significant comments!")
    return issues

# Check for issues
issues = validate_dataset(all_samples)
if issues:
    print("\n--- Data Issues Found ---")
    for issue in issues:
        print(issue)
else:
    print("\nNo data issues found.")

# Save to JSON
with open(OUTPUT_FILENAME, "w") as f:
    json.dump(all_samples, f, indent=2)

# Summary statistics
num_rewarded = sum(1 for s in all_samples if s["surprise_unlocked"])
total_samples = len(all_samples)
print(f"Generated {total_samples} samples to {OUTPUT_FILENAME}")
print(f"Number of rewarded samples: {num_rewarded} ({num_rewarded / total_samples * 100:.2f}%)")

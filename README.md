# 🚀 Karma AI Reward System

The **Karma Surprise Box** is an AI-driven reward microservice built with Python, FastAPI, and Docker. It evaluates user activity metrics (e.g., login streaks, posts created, upvotes received) and awards karma points based on predefined rules or a random mystery reward. The system assigns rarities (common, rare, legendary) to rewards and includes validation to reject invalid inputs (e.g., negative metrics).

This project is designed to incentivize user engagement on a platform by delivering surprise rewards, logged and tracked via a SQLite database.

---

## ✨ Features

- **Rule-Based Rewards**: Awards karma based on user activity conditions with varying rarities.
- **Mystery Reward**: A 10% chance of a random common reward.
- **Validation**: Rejects inputs with negative metrics.
- **API Endpoints**: Health check, version info, and reward evaluation.
- Dockerized: Easy deployment and scalability.

---

## ⚙️ Prerequisites

- Python 3.9+
- Docker
- Git (for version control)

---

## 🛠️ Tech Stack

| Component     | Technology       |
|---------------|------------------|
| Language      | Python 3.9        |
| Backend       | FastAPI           |
| Packaging     | Docker (471MB image) |
| Database      | SQLite            |
| Model  | Random Forest (scikit-learn) |

## 🧑‍💻 Installation

### 🔁 Clone the Repository

```bash
git clone https://github.com/your-username/karma-surprise-box.git
cd karma-surprise-box
```
### Docker Deployment
## Build the Docker Image

Build the Docker Image
```bash
docker build -t karma-box .
```
Run the Container
```bash
docker run -p 8000:8000 karma-box
```
### API Endpoints

| Endpoint              | Method | Description                            |
| --------------------- | ------ | -------------------------------------- |
| `/health`             | GET    | Returns `{"status": "ok"}`             |
| `/version`            | GET    | Returns model version and feature info |
| `/check-surprise-box` | POST   | Evaluates metrics and returns a reward |


## Request Format

```json
{
  "user_id": "string",
  "date": "YYYY-MM-DD",
  "daily_metrics": {
    "login_streak": 5,
    "posts_created": 2,
    "comments_written": 4,
    "upvotes_received": 10,
    "quizzes_completed": 1,
    "buddies_messaged": 3,
    "karma_spent": 20,
    "karma_earned_today": 40
  }
}
```
## Response Format
``` json
{
  "user_id": "string",
  "surprise_unlocked": true,
  "reward_karma": 50,
  "reason": "High engagement score",
  "rarity": "rare",
  "box_type": "rule-based",
  "status": "delivered"
}
```
## 🧠 Conditions & Logic
- System checks metrics against config.json rules.

- The highest matching rule is applied.

- If no rule matches, a 10% mystery reward is triggered.

## Reward Rules Table


| **Condition**                                                              | **Reward Karma** | **Reason**                                      | **Rarity** | **Box Type**        |
| -------------------------------------------------------------------------- | ---------------- | ----------------------------------------------- | ---------- | ------------------- |
| `login_streak >= 3` and `posts_created >= 1` and `quizzes_completed >= 1`  | 10               | Great engagement streak with posts and quizzes! | common     | streak              |
| `upvotes_received >= 5`                                                    | 15               | Good upvotes received today!                    | rare       | engagement          |
| `karma_spent >= 15`                                                        | 12               | Generous karma spending!                        | common     | social              |
| `login_streak >= 5`                                                        | 10               | Consistent daily logins!                        | common     | streak\_engager     |
| `quizzes_completed >= 2`                                                   | 12               | Good quiz participation!                        | common     | quiz\_enthusiast    |
| `posts_created >= 2` and `upvotes_received >= 3`                           | 18               | Prolific posting with solid feedback!           | rare       | content\_creator    |
| `buddies_messaged >= 3` and `comments_written >= 2`                        | 15               | Active community engagement!                    | rare       | community\_champion |
| `login_streak >= 10` and `quizzes_completed >= 5` and `posts_created >= 3` | 25               | Legendary engagement streak!                    | legendary  | legendary\_streak   |

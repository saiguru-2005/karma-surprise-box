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

-


# Modular Recommender

## ⚙️ Setup

### 1. Create environment
conda env create -f env/environment.yml

### 2. Activate environment
conda activate rec-sys-base

### 3. Install dev tools (pre-commit hooks)
make setup

## 🧪 Development Workflow
### Run tests
make test

### Format code
make format

### Run lint checks
make lint


## 🔐 Security

This project uses:

pre-commit hooks

gitleaks (for detecting secrets)

If a commit fails due to secret detection:

Remove the secret or move it to .env

Stage changes again

Commit

## 📊 Experiment Tracking

### Run mlflow
make mlflow

### Run TensorBoard
make tensorboard

## 🧠 Notes

Do NOT commit .env files

Secrets are automatically scanned before commit

## Run full scan manually
pre-commit run --all-files

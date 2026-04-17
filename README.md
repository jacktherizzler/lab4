# Lab 4: Automated Model Deployment Using GitHub Actions (CI/CD)

This repository automates model training and deployment for wine quality inference.
On every push to `main`, GitHub Actions trains a model, compares the new metrics against
repository baseline variables, and only pushes a Docker image to Docker Hub when both
metrics improve.

## Repository Structure

```text
lab4/
├── .github/workflows/train-and-deploy.yml
├── app/
│   ├── __init__.py
│   └── main.py
├── data/
├── model/
├── outputs/
├── scripts/
│   └── train.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## Required GitHub Secrets

Add these repository secrets before expecting the deploy job to publish an image:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `GH_REPO_PAT`

## Required GitHub Variables

Set these repository variables in the GitHub repo:

- `BEST_MSE`
- `BEST_R2`

Recommended initial values:

- `BEST_MSE = 999999`
- `BEST_R2 = -999999`

## Workflow Behavior

### Job 1: `train`

- sets up Python 3.11
- installs dependencies
- runs `scripts/train.py`
- saves `outputs/trained_model.pkl`
- saves `outputs/results.json`
- uploads both files as GitHub artifacts

### Job 2: `deploy`

- runs only after `train` succeeds on a push to `main`
- downloads the trained model artifact
- compares current `MSE` and `R2 Score` against `BEST_MSE` and `BEST_R2`
- prints `2022BCD0002----Metric did not improve` when the model is worse or unchanged
- logs in to Docker Hub when the model improves
- builds and pushes the Docker image
- updates `BEST_MSE` and `BEST_R2` in GitHub repository variables

## Docker Image Naming

The workflow pushes:

- `<DOCKERHUB_USERNAME>/wine_predict_2022BCD0002:latest`
- `<DOCKERHUB_USERNAME>/wine_predict_2022BCD0002:<github_sha>`

## Local Validation

### Train locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py
cp outputs/trained_model.pkl model/trained_model.pkl
cp outputs/results.json model/results.json
```

### Run the API locally

```bash
uvicorn app.main:app --reload
```

### Test inference

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
```

Expected response format:

```json
{
  "name": "SriHarsha Bodicherla",
  "roll_no": "2022BCD0002",
  "wine_quality": 5
}
```

## Deliverables

Submit:

1. GitHub repository link
2. Docker Hub repository link
3. Screenshots showing:
   - successful workflow with both train and deploy jobs
   - workflow where train runs and deploy is skipped
   - Docker Hub image repository
   - pulled image running locally
   - successful inference response


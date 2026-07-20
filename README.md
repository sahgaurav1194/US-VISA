# US Visa Approval Prediction — End-to-End ML Pipeline

Modular machine learning pipeline that predicts US visa approval outcomes
(**Certified / Denied**), containerised with Docker and deployed to AWS EC2
via Amazon ECR with a fully automated GitHub Actions CI/CD pipeline.

## Problem Statement
OFLC processes thousands of visa applications every year. The goal is to
predict, from applicant and employer attributes, whether a case will be
certified — so that likely approvals can be shortlisted faster.

## Architecture

```
MongoDB Atlas
     |
Data Ingestion  -->  Data Validation  -->  Data Transformation  -->  Model Trainer
     |                    |                       |                       |
feature store        schema report      preprocessor + SMOTEENN     best model (RF / LogReg / DT)
                                                                          |
                                                        FastAPI  <--  model.pkl
                                                           |
                                        Docker --> Amazon ECR --> EC2 (self-hosted runner)
```

## Tech Stack
- **Python**, Pandas, NumPy, Scikit-learn, imbalanced-learn
- **Models:** Logistic Regression, Random Forest, Decision Tree (GridSearchCV tuned)
- **Data:** MongoDB Atlas
- **Serving:** FastAPI + Jinja2 form UI
- **MLOps:** Docker, Amazon ECR, AWS EC2, GitHub Actions (CI/CD with self-hosted runner)
- **Environment:** conda + requirements.txt

## Project Structure
```
us_visa/
├── components/        # ingestion, validation, transformation, trainer
├── configuration/     # MongoDB connection
├── data_access/       # DataFrame export from MongoDB
├── entity/            # config / artifact dataclasses, model wrapper
├── pipeline/          # training + prediction pipelines
├── utils/             # yaml / object / numpy helpers
├── logger.py          # timestamped run logs
└── exception.py       # custom exception with file + line info
```

## How to Run

```bash
conda create -n visa python=3.10 -y
conda activate visa
pip install -r requirements.txt

# set your MongoDB Atlas connection (optional — falls back to notebooks/EasyVisa.csv)
export MONGODB_URL="mongodb+srv://<user>:<password>@cluster.mongodb.net"

python demo.py     # train
python app.py      # serve UI on http://localhost:8080
```

## CI/CD — GitHub Actions → ECR → EC2
Every push to `main`:
1. **CI** — checkout, lint, tests
2. **CD** — build Docker image and push to Amazon ECR
3. **Deployment** — self-hosted EC2 runner pulls the latest image and
   relaunches the container with zero manual intervention

**IAM (least privilege):** `AmazonEC2FullAccess`, `AmazonEC2ContainerRegistryFullAccess`

**GitHub Secrets required:**
`AWS_ACCESS_KEY_ID` · `AWS_SECRET_ACCESS_KEY` · `AWS_DEFAULT_REGION` · `ECR_REPO` · `MONGODB_URL`

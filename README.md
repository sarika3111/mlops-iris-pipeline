# mlops-iris-pipeline

Assignment Breakdown: MLOps Pipeline with Iris Dataset
________________________________________
Part 1: Repository and Data Versioning (4 marks)
Steps:
1.	Created a GitHub Repo:
Directory Structure:
mlops-iris-pipeline/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── api/
├── docker/
├── .github/workflows/
├── requirements.txt
└── README.md
>> Download Dataset:
•	Use: Iris Dataset CSV
•	Save to data/raw/iris.csv
>> Preprocess Data:
•	Load with pandas
•	Assign column names
•	Encode target labels
•	Save to data/processed/iris_clean.csv
>> Version Control:
•	Use Git for code
•	DVC is optional (not required for Iris due to small size)
________________________________________
Part 2: Model Development & Experiment Tracking (6 marks)
Steps:
1.	Train Two Models:
•	Logistic Regression
•	Random Forest
2.	Use MLflow:
•	Track parameters, metrics (accuracy, precision, recall)
•	Log models and artifacts
3.	Model Selection:
•	Choose best model based on metrics
•	Register it in MLflow Model Registry
4.	Save Final Model:
•	Save to models/final_model.pkl
________________________________________
Part 3: API & Docker Packaging (4 marks)
Steps:
1.	Create API:
•	Use FastAPI or Flask
•	Endpoint: /predict
•	Accept JSON input: feature values
•	Return: predicted class
2.	Dockerize:
•	Write Dockerfile
•	Build image: docker build -t iris-api .
•	Run container: docker run -p 8000:8000 iris-api
________________________________________
Part 4: CI/CD with GitHub Actions (6 marks)
Steps:
1.	Create Workflow:
•	File: .github/workflows/main.yml
2.	Pipeline Tasks:
•	Lint code with flake8
•	Run tests with pytest
•	Build Docker image
•	Push to Docker Hub
•	Deploy locally or to EC2
________________________________________
Part 5: Logging & Monitoring
Steps:
1.	Logging:
•	Use Python logging module
•	Log incoming requests and predictions
•	Save to file or SQLite DB
2.	Monitoring (Optional):
•	Add /metrics endpoint
•	Use Prometheus to scrape metrics
•	Create Grafana dashboard

---------------------------------------- 📘 End‑to‑End Machine Learning Project using AWS SageMaker  ----------------------------------------------------
📌 Project Overview
This project demonstrates how to build a complete machine learning pipeline using Amazon SageMaker.

The goal of this project is to train a Linear Regression model to predict student exam scores based on the number of study hours.

The project covers:

Data preprocessing

Train–test splitting

Uploading data to Amazon S3

Model training using SageMaker Linear Learner

Saving trained model artifacts

--------------------------------------  🛠️ Technologies Used -----------------------------------------
Amazon SageMaker

Amazon S3

AWS IAM Role

Python

Pandas & NumPy

Scikit‑learn

Boto3

------------------------------------------ 🧱 Project Workflow (Step‑by‑Step) -----------------------------------
# Step 1: Setting Up SageMaker Environment # 
Logged into AWS Console

Opened Amazon SageMaker

Created SageMaker Studio (single-user setup)

Launched JupyterLab environment
images/<img width="1919" height="1026" alt="SC37E1~1" src="https://github.com/user-attachments/assets/6259049a-2225-422f-b99a-d2a5a8ec2658" />
images/02-studio-setup.png

# Step 2: Loading the Dataset #
Loaded student_scores.csv using Pandas

Dataset contains:

Hours studied

Scores obtained

df = pd.read_csv("student_scores.csv")
📸 Add Screenshot:
images/03-dataset-preview.png

# Step 3: Data Preprocessing #
Separated features (Hours) and label (Scores)

Converted data into float32 format (required by SageMaker)

Split dataset:

80% training data

20% testing data

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
📸 Add Screenshot:
images/04-train-test-split.png

# Step 4: Convert Data to SageMaker Format #
SageMaker Linear Learner requires data in RecordIO protobuf format.

Converted NumPy arrays into SageMaker format

Used in-memory buffer

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, X_train, y_train)
buf.seek(0)
This prepares data for SageMaker training.

Step 5: Upload Data to Amazon S3
Created S3 bucket

Uploaded training and testing data

Defined S3 paths

boto3.resource("s3") \
    .Bucket(bucket_name) \
    .Object(f"{prefix}/train/student-data") \
    .upload_fileobj(buf)
📸 Add Screenshot:
images/05-s3-upload.png

# Step 6: Retrieve Linear Learner Algorithm #
Used AWS built‑in Linear Learner container:

container = sagemaker.image_uris.retrieve(
    "linear-learner",
    boto3.Session().region_name
)
This loads AWS‑managed Docker image for training.

# Step 7: Create SageMaker Estimator #
Configured:

Instance type: ml.c4.xlarge

Instance count: 1

Output path: S3 location

linear = sagemaker.estimator.Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type="ml.c4.xlarge",
    output_path=output_location,
    sagemaker_session=session
)
# Step 8: Set Hyperparameters #
Configured model settings:

feature_dim = 1

predictor_type = regressor

epochs = 6

mini_batch_size = 4

linear.set_hyperparameters(
    feature_dim=1,
    predictor_type="regressor",
    mini_batch_size=4,
    epochs=6
)
# Step 9: Train the Model #
Started training job:

linear.fit({"train": s3_train_data})
What SageMaker does:

Launches ML instance

Downloads training data

Trains Linear Regression model

Stores model artifacts in S3

📸 Add Screenshot:
images/06-training-job-running.png

# Step 10: Training Completed #
Model successfully trained

Output model stored in S3

Ready for deployment

📸 Add Screenshot:
images/07-training-complete.png

--------------✅ Final Outcome ------------
✔ Successfully built an end‑to‑end ML pipeline
✔ Preprocessed and uploaded data to S3
✔ Trained Linear Regression model using SageMaker
✔ Model artifacts stored in S3
✔ Gained hands‑on experience with AWS ML workflow

-------------- 📌 What I Learned ------------
How SageMaker training jobs work

How to use S3 for ML data storage

How to configure estimators and hyperparameters

How cloud‑based ML pipelines operate

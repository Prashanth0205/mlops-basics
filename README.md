# MLFlow on DagsHub:

import dagshub
dagshub.init(repo_owner='Prashanth0205', repo_name='mlops-basics', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

# MLFlow on AWS

## MLFlow on AWS setup:

1. Login to AWS console 
2. Create IAM user with AdminstratorAccess 
3. Export the credentials in your AWS CLI by running "aws configure"
4. Create a S3 bucket 
5. Create EC2 Machine and add security groups 5000 port 

Run the following command on EC2 Machine 
'''bash 
sudo apt update 
sudo apt install python3-pip 
sudo pip3 install pipenv
sudo pip3 install virtualenv 
mkdir mlflow 
'''
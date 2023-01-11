from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
import mlflow
from datetime import datetime
import random

def init_ml_flow_tracking(**kwargs):
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    
    ml_flow_tracking_url = Variable.get("mlFlowTrackingURL")
    mlflow.set_tracking_uri(ml_flow_tracking_url)
    mlflow.set_experiment(Variable.get("mlFlowExperimentName"))
    #mlflow.set_tag("mlflow.runName", "custom_ner_run_"+timestampStr)
    return init_ml_flow_tracking

def train_model(**kwargs):
    models = ["Bidirectional LSTM", "CRF", "Transformer", "RoBERTa", "Flair"]
    ml_flow_tracking_url = Variable.get("mlFlowTrackingURL")
    mlflow.set_tracking_uri(ml_flow_tracking_url)
    mlflow.set_experiment(Variable.get("mlFlowExperimentName"))
    # Iterate over the list of models
    for model in models:
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")
        with mlflow.start_run(run_name=model+"_run_"+timestampStr):
            # Set the run parameters
            mlflow.log_param("model_type", model)
            mlflow.log_param("learning_rate", random.uniform(0.001, 0.01))
            mlflow.log_param("dropout_rate", random.uniform(0.1, 0.5))
            mlflow.log_param("batch_size", random.randint(16, 64))
            mlflow.log_param("epochs", random.randint(10, 20))
            mlflow.log_param("ner_type", "custom")
            
            metrics = {
                "accuracy": random.uniform(0.5, 0.9),
                "precision": random.uniform(0.5, 0.9),
                "recall": random.uniform(0.5, 0.9),
                "f1_score": random.uniform(0.5, 0.9)
            }
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            mlflow.set_tag("mlflow.model.name", model)
            # Set the run tags
            mlflow.set_tag("author", "Abhi")        
    

with DAG(dag_id="custom_ner_pipeline", start_date=days_ago(1),tags=["Demo"]) as dag:
    start = DummyOperator(task_id="start")
    
    dataStore_id = "Connect_to_" + Variable.get("dataStore")
    
    tag_dvc = BashOperator(task_id='tag_dvc', bash_command='return 1')
   
    with TaskGroup("data_load", tooltip="Data Loading") as section_1:
        task_1 = BashOperator(task_id="Fetch_DataStore_Keys", bash_command=':')
        task_2 = BashOperator(task_id=dataStore_id, bash_command='sleep 1')
        task_3 = BashOperator(task_id='load_data', bash_command='sleep 3')
        
        task_1 >> task_2 >> task_3 
    
    section_1 >> tag_dvc

    with TaskGroup("data_preparation", tooltip="task group #2") as section_2:        
        task_4 = BashOperator(task_id="seed", bash_command=':')
        task_5 = BashOperator(task_id="data_preparation_job", bash_command='sleep 5')
        task_6 = BashOperator(task_id="train_test_split", bash_command='sleep 1')
        task_4 , task_5 >> task_6
        
    
    
    task_init_mlflow = PythonOperator(task_id='init_mlflow', python_callable= init_ml_flow_tracking, op_kwargs = {"x" : "Apache Airflow"}, dag=dag)
    
    task_train_model = PythonOperator(task_id='train_model', python_callable= train_model, op_kwargs = {"x" : "Apache Airflow"}, dag=dag)
    
   

    end = DummyOperator(task_id='end')

    
    start >> section_1 >>  section_2 >> task_init_mlflow >> task_train_model >> end

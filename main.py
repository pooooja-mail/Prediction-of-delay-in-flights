
from pyspark.sql import SparkSession

from pyspark.sql.functions import lit
from pyspark.sql.functions import when

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import MultilayerPerceptronClassifier

from sparkmeasure import StageMetrics
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import shutil


DATASET = "dataset/DelayedFlights.csv"
OUTPUT_DIR = './output'
# Arrival delay pretty much 100% correlates with departure delay, without it it's impossible to predict anything.
USE_ARRIVAL_DELAY = False
DELAY_THRESHOLD = 15.0


def pipeline(spark, model_type, dataset_size):

    df = spark.read.format("csv").load(DATASET, header=True, inferSchema=True)
    max_size = df.count()
    if dataset_size < max_size:
        df = df.sample(dataset_size / max_size)

    # drop unnecessary columns
    drop_columns = ["_c0", "Year", "FlightNum", "TailNum", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "CancellationCode"]
    if not USE_ARRIVAL_DELAY:
        drop_columns.append("ArrDelay")
    df = df.drop(*drop_columns)

    # add binary indicator for delay above threshold
    df = df.withColumn("DepDelayBinary", when((df.DepDelay > DELAY_THRESHOLD), lit(1)).otherwise(lit(0)))

    #df.show()

    # Separate categorical and real values features
    categorical_string_features = ["UniqueCarrier", "Origin", "Dest"]
    categorical_indexed_features = ["Cancelled", "Diverted"]
    categorical_features = categorical_string_features + categorical_indexed_features
    real_value_features = df.columns
    for categorical_feature in categorical_features:
        real_value_features.remove(categorical_feature)
    real_value_features.remove("DepDelay")
    real_value_features.remove("DepDelayBinary")


    # Fill missing values in with mean
    imputer = Imputer(inputCols=real_value_features, outputCols=real_value_features)
    imputer = imputer.fit(df)
    df = imputer.transform(df)

    # turn real valued features into vector
    real_value_assembler = VectorAssembler(inputCols=real_value_features, outputCol='real_value_features')
    df = real_value_assembler.transform(df)

    # scale real valued features into normal distributions (subtract off mean then divide by standard deviation)
    scalar = StandardScaler(inputCol="real_value_features", outputCol="real_value_features_scaled", withMean=True, withStd=True)
    scalar = scalar.fit(df)
    df = scalar.transform(df)

    # Categorical Features
    categorical_string_features_indicies = []
    categorical_string_features_one_hot = []
    for string_feature in categorical_string_features:
        categorical_string_features_indicies.append(string_feature + "_index")
        categorical_string_features_one_hot.append(string_feature + "_one_hot")

    for index_feature in categorical_indexed_features:
        categorical_string_features_one_hot.append(index_feature + "_one_hot")

    indexer = StringIndexer(inputCols=categorical_string_features, outputCols=categorical_string_features_indicies)
    indexer = indexer.fit(df)
    df = indexer.transform(df)

    # Categorical -> one hot
    one_hot = OneHotEncoder(inputCols=categorical_string_features_indicies + categorical_indexed_features, outputCols=categorical_string_features_one_hot)
    one_hot = one_hot.fit(df)
    df = one_hot.transform(df)

    # combine all into one vector
    final_assembler = VectorAssembler(inputCols=["real_value_features_scaled"] + categorical_string_features_one_hot, outputCol="all_features")
    df = final_assembler.transform(df)

    # split into train and test
    train, test = df.randomSplit([0.7, 0.3])


    def get_accuracy(model, data):
        predict = model.transform(data)
        return  predict.filter(predict.DepDelayBinary == predict.prediction).count() / float(predict.count())


    if model_type == 'logistic_regression':
        # model: logistic regression
        logistic_regression = LogisticRegression(featuresCol="all_features", labelCol="DepDelayBinary", regParam=0.1, elasticNetParam=1.0)
        logistic_regression = logistic_regression.fit(train)
        train_accuracy = get_accuracy(logistic_regression, train)
        test_accuracy = get_accuracy(logistic_regression, test)

    elif model_type == 'neural_network':

        input_size = len(real_value_features)
        df_pandas =  df.toPandas()
        for cat_feature in categorical_features:
            input_size += (df_pandas[cat_feature].drop_duplicates().shape[0] - 1)
            
        # model: neural network classifier
        neural_network = MultilayerPerceptronClassifier(featuresCol="all_features", labelCol="DepDelayBinary", layers = [input_size,32,2], maxIter=50, blockSize=256, solver="gd")
        neural_network = neural_network.fit(train)
        train_accuracy = get_accuracy(neural_network, train)
        test_accuracy = get_accuracy(neural_network, test)

    return train_accuracy, test_accuracy




def collect_metrics(model_type, dataset_size):
    # start instance
    spark = SparkSession.builder.master("local").appName("FlightDelays").config("spark.ui.port", "4050").config("spark.driver.memory", "15g").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # run Spark workload with instrumentation
    stagemetrics = StageMetrics(spark)
    stagemetrics.begin()
    train_accuracy, test_accuracy = pipeline(spark, model_type, dataset_size)
    stagemetrics.end()

    output_path = f'{OUTPUT_DIR}/metrics_model_{model_type}_dataset_size_{dataset_size}'
    #print(output_path)

    # save session metrics data in json format (default)
    df = stagemetrics.create_stagemetrics_DF("PerfStageMetrics")
    stagemetrics.save_data(df.orderBy("jobId", "stageId"), output_path)

    spark.stop()

    metrics_json = get_metrics_json(output_path)
    metrics_json['train_accuracy'] = train_accuracy
    metrics_json['test_accuracy'] = test_accuracy
    return metrics_json


def get_metrics_json(dir):
    # get metrics from json file saved to disk (output of sparkmeasure)
    duration = 0
    for file_name in os.listdir(dir):
        if file_name[-4:] == "json":
            file = open(f'./{dir}/{file_name}')
            lines = file.readlines()
            for line in lines:
                json_data = json.loads(line)
                duration += json_data['stageDuration']
            file.close()
    
    duration /= 1000 # ms -> s
    return {'duration': duration}


def plot_time(model, dataset_test_sizes, durations):
    plt.plot(dataset_test_sizes, durations)
    plt.title(f"Pipeline Compute Time ({model})")
    plt.xlabel('Dataset Size (rows)')
    plt.ylabel('Compute Time (seconds)')

    plt.ylim(bottom=0)
    plt.xscale("log")


    plt.savefig(f"{OUTPUT_DIR}/dataset_size_vs_duration_{model}.png")
    plt.clf()


def plot_accuracy(model, dataset_test_sizes, train_accuracies, test_accuracies):
    plt.plot(dataset_test_sizes, train_accuracies, color='blue', label='train')
    plt.plot(dataset_test_sizes, test_accuracies, color='green', label='test')
    plt.title(f"Model Accuracy ({model})")
    plt.xlabel('Dataset Size (rows)')
    plt.ylabel('Accuracy')

    plt.ylim(bottom=0, top=1)
    plt.xscale("log")
    plt.legend()

    plt.savefig(f"{OUTPUT_DIR}/dataset_size_vs_accuracy_{model}.png")
    plt.clf()


def main():

    # experiment to collect metrics on time and accuracy given different dataset sizes.

    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    model_types = ['logistic_regression', 'neural_network']
    #dataset_test_sizes = [50, 100, 200, 350, 500, 750, 1000, 2500, 5000, 10000, 50000, 100000]
    dataset_test_sizes = [100, 1000, 10000, 50000, 100000, 500000, 1000000, 2000000]

    for model_type in model_types:
        durations = []
        train_accuracies = []
        test_accuracies = []
        print(model_type)
        for dataset_size in dataset_test_sizes:
            metrics = collect_metrics(model_type, dataset_size)
            durations.append(metrics['duration'])
            train_accuracies.append(metrics['train_accuracy'])
            test_accuracies.append(metrics['test_accuracy'])
            print(dataset_size)
            print(metrics['duration'], metrics['train_accuracy'], metrics['test_accuracy'])
        plot_time(model_type, dataset_test_sizes, durations)
        plot_accuracy(model_type, dataset_test_sizes, train_accuracies, test_accuracies)



if __name__ == "__main__":
    main()

from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, lit, udf, col
from pyspark.sql.types import ArrayType, FloatType, IntegerType
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import DenseVector, VectorUDT

import cv2

import numpy as np


def createSparkSession():
    return SparkSession.builder \
        .appName("Futurama") \
        .master("local[*]") \
        .getOrCreate()

def readData(spark: SparkSession):
    labels_df = spark.read.csv("data/train_data.csv", header=True, inferSchema=True, sep=',')
    train_images_df = spark.read.format("binaryFile").load("data/train_img/*.png")
    test_images_df = spark.read.format("binaryFile").load("data/test_img/*.png")

    return labels_df, train_images_df, test_images_df

def extractFilenameFromPath(train_df, test_df):
    train_df = train_df.withColumn("filename", regexp_extract("path", r"([^/]+)$", 1))
    test_df = test_df.withColumn("filename", regexp_extract("path", r"([^/]+)$", 1))

    return train_df, test_df

def joinLabels(train_df, labels_df):
    train_df = train_df.join(labels_df, train_df.filename == labels_df.file)

    return train_df

def extract_features(image_data):
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    return hist.flatten().tolist()

def array_to_vector(array):
    return DenseVector(array)

def trainBinaryModel(train_images_df, characters):
    models = {}

    for character in characters:
        # Convert label to Integer
        train_images_df = train_images_df.withColumn(character, train_images_df[character].cast(IntegerType()))

        # Setup and train a binary classifier
        rf = RandomForestClassifier(featuresCol="features", labelCol=character, numTrees=10)
        model = rf.fit(train_images_df)

        models[character] = model

    return models


if __name__ == "__main__":
    spark = createSparkSession()
    characters = ["isFry", "isLeela", "isBender"]

    labels_df, train_images_df, test_images_df = readData(spark)
    train_images_df, test_images_df = extractFilenameFromPath(train_images_df, test_images_df)
    train_images_df = joinLabels(train_images_df, labels_df)

    # Extract features using color histogram
    extract_features_udf = udf(extract_features, ArrayType(FloatType()))

    train_images_df = train_images_df.withColumn("features_array", extract_features_udf("content"))
    test_images_df = test_images_df.withColumn("features_array", extract_features_udf("content"))

    # Convert the list to a VectorUDT in a new column 'features'
    array_to_vector_udf = udf(array_to_vector, VectorUDT())

    train_images_df = train_images_df.withColumn("features", array_to_vector_udf("features_array"))
    test_images_df = test_images_df.withColumn("features", array_to_vector_udf("features_array"))

    # Add label cols to test dataframe
    for col_name in characters:
        test_images_df = test_images_df.withColumn(col_name, lit(0))

    # Train binary model for each character
    models = trainBinaryModel(train_images_df, characters)

    # Evaluate the model on the test set
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

    # Make predictions using test images df
    for character, model in models.items():
        # Make predictions for the current character
        predictions = model.transform(test_images_df)

        # Assign prediction to the corresponding column
        predictions = predictions.withColumnRenamed("prediction", f"prediction_{character}")

        # Evaluate precision for the current character
        evaluator.setPredictionCol(f"prediction_{character}")
        evaluator.setLabelCol(character)
        accuracy = evaluator.evaluate(predictions)
        print(f"Precision for {character}: {accuracy:.2f}")

        test_images_df = test_images_df.join(predictions.select("path", f"prediction_{character}"), on="path",
                                             how="inner")

    # Prepare results
    test_images_df = (test_images_df
        .drop("isFry", "isLeela", "isBender")
        .withColumnRenamed("prediction_isFry", "isFry")
        .withColumnRenamed("prediction_isLeela", "isLeela")
        .withColumnRenamed("prediction_isBender", "isBender")
        .withColumnRenamed("filename", "file")
        .select("file", "isLeela", "isFry", "isBender")
    )

    test_images_df = test_images_df.repartition(1)

    # Write results
    test_images_df.write.mode("overwrite").option("header", "true").csv("submission")

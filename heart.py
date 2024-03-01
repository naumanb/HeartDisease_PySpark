
from pyspark.sql import SparkSession
from structs import heart_schema


spark = SparkSession.builder.getOrCreate()

heart_df = spark.read.csv("heart_data.csv", schema=heart_schema, header=True) # Reading the csv file and inferring the schema

heart_df.printSchema()
print(heart_df.schema)
heart_df.show() # Showing the data

heart_df.createOrReplaceTempView("HEART_DATA") # Creating a temporary view from the dataframe

df = spark.sql("SELECT * from HEART_DATA") # Selecting all the data from the view
df = df.drop('heart_disease') # Dropping the heart disease column
df = df.drop('index') # Dropping the index column
df.printSchema() 

inputList = df.columns # Getting the column names
print(inputList)

from pyspark.ml.feature import VectorAssembler # Importing the VectorAssembler

features_assembler = VectorAssembler(inputCols = inputList, outputCol = 'features') # Assembling the features

df = features_assembler.transform(heart_df) # Transforming the dataframe

df.printSchema()

working_df = df.select('features', 'heart_disease') # Selecting the features and the heart disease (dependent variable) column

working_df.show(10)

training, test = working_df.randomSplit([0.7, 0.3]) # Splitting the data into training and test
from pyspark.ml.regression import LinearRegression # Importing the Linear Regression model

lr = LinearRegression(featuresCol = 'features', labelCol = 'heart_disease') # Creating the model

model = lr.fit(training) # Fitting the model
predict_output = model.transform(test) # Transforming the test data

predict_output.show()

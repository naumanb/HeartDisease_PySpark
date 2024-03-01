from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.getOrCreate()

cust_df = spark.read.csv("customer_data.csv", header=True, inferSchema=True)

cust_df.printSchema()

cust_df.createOrReplaceTempView("CUST_DATA")

df = spark.sql("SELECT income, spending from CUST_DATA")

inputList = df.columns

features_assembler = VectorAssembler(inputCols = inputList, outputCol = 'features')

df = features_assembler.transform(cust_df)
df.show(10)

working_df = df.select('features', 'purchase_frequency')

training, test = working_df.randomSplit([0.8, 0.2])
lr = LinearRegression(featuresCol = 'features', labelCol = 'purchase_frequency')

model = lr.fit(training)

predict_output = model.transform(test)

predict_output.show()

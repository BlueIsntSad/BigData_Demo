from utils import _initialize_spark
from pyspark.sql.types import *
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import SQLTransformer

import streamlit as st

spark, _ = _initialize_spark()

st.write("# :tada: Hello Pyspark")

st.subheader("2 ways of creating Dataframes")
rdd = spark.sparkContext.parallelize([('1', 'a'), ('2', 'b'), ('3', 'c'), ('4', 'd'), ('5', 'e'), ('6', 'f')])
schema = StructType([StructField('ID', StringType(), True), StructField('letter', StringType(), True)])
df = spark.createDataFrame(rdd, schema)

st.write(df.toPandas())

titanic = spark.read.format(
    'csv'
).option(
    'header', 'true'
).option(
    'inferSchema', 'true'
).load("titanic.csv")

st.write(titanic.toPandas())

st.subheader("Interacting with Dataframe")
st.write(titanic[titanic['Survived'] == 1].toPandas())

titanic.createOrReplaceTempView('titanic')
st.write(spark.sql('SELECT * FROM titanic WHERE Survived = 1').toPandas())

st.subheader("Machine Learning")
mean_age = spark.sql('SELECT MEAN(Age) FROM titanic').collect()[0][0]
titanic = titanic.na.fill(mean_age, ['Age'])
binarizer = Binarizer(threshold=50, inputCol="Age", outputCol="binarized_age")
titanic_bin = binarizer.transform(titanic)
st.write(titanic_bin.toPandas())

regex='"(Mr)"'
sqlTrans = SQLTransformer(
    statement=f"SELECT *, regexp_extract(Name, {regex}) AS Civility FROM __THIS__"
)
st.write(sqlTrans.transform(titanic).toPandas())
from pyspark.rdd import RDD
from pyspark.sql import Row
import streamlit as st
from utils import _initialize_spark

st.write("# :tada: Hello Pyspark")

spark, sc = _initialize_spark()
st.write("[Link to Spark window](http://localhost:4040)")

st.write("## Create RDD from a Python list")

l = list(range(10))
# st.write(l)

rdd = sc.parallelize(l)
rdd.cache()
st.write(rdd)

st.write("## Get results through actions")
st.write(rdd.collect())
st.write(rdd.take(3))
st.write(rdd.count())

st.write("## Transform RDDs")
st.write(rdd.filter(lambda x: x%2==0).collect())  # talk about lazy evaluation here: filter still non evaluated rdd
st.write(rdd.map(lambda x: x*2).collect()) 
st.write(rdd.map(lambda x: x*2).reduce(lambda x, y: x + y))  # reduce runs all previous rdds
# Compare the two following
st.write(rdd.map(lambda x: list(range(x))).collect()) 
st.write(rdd.flatMap(lambda x: list(range(x))).collect()) 

st.write("## Wordcount")
file_rdd = sc.textFile("lorem.txt")
st.write(file_rdd.collect())  # so what's inside ?
st.write(file_rdd.flatMap(lambda sentence: sentence.split()).map(lambda word: (word, 1)).reduceByKey(lambda x,y: x+y).collect())
st.write()


# spark.stop()
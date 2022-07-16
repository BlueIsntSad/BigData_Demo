import sys
import pytz
import datetime
from datetime import datetime
from pyspark import SparkConf
from pyspark.sql import SparkSession

def currentTime(timezone = 'Asia/Ho_Chi_Minh') -> datetime:
    ''' Thời gian hiện tại với mặc định timezone Tp.HoChiMinh '''
    return datetime.now(pytz.timezone(timezone))

def _initialize_spark() -> SparkSession:
    """Create a Spark Session for Streamlit app"""
    conf = SparkConf()\
        .set('spark.sql.legacy.timeParserPolicy', 'LEGACY')\
        .setAppName("bigdata").setMaster("local")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark, spark.sparkContext
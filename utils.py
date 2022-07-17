import sys
import re
import pytz
import datetime
from datetime import datetime
from pyspark import SparkConf
from pyspark.sql import SparkSession

def currentTime(timezone = 'Asia/Ho_Chi_Minh') -> datetime:
    ''' Thời gian hiện tại với mặc định timezone Tp.HoChiMinh '''
    return datetime.now(pytz.timezone(timezone))

def validateURL(url):
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    if re.match(regex, url) is not None:
        return True
    else:
        return False

def _initialize_spark() -> SparkSession:
    """Create a Spark Session for Streamlit app"""
    conf = SparkConf()\
        .set('spark.sql.legacy.timeParserPolicy', 'LEGACY')\
        .setAppName("bigdata").setMaster("local")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark, spark.sparkContext
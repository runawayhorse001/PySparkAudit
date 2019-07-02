from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark regression example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


from PySparkAudit import dtypes_class, hist_plot, bar_plot, freq_items,feature_len
from PySparkAudit import dataset_summary, rates
from PySparkAudit import trend_plot, auditing
# path = '/home/feng/Desktop'


data = spark.read.csv(path='Heart.csv',
                      sep=',', encoding='UTF-8', comment=None, header=True, inferSchema=True)

print(auditing(data, display=True))

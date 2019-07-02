from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark regression example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


# import PySpark Audit functions
from PySparkAudit import data_types, hist_plot, bar_plot, freq_items,feature_len
from PySparkAudit import dataset_summary, rates
from PySparkAudit import trend_plot, auditing

# load dataset
data = spark.read.csv(path='Heart.csv',
                      sep=',', encoding='UTF-8', comment=None, header=True, inferSchema=True)

# audit function by function

# data types
print(data_types(data))

# check frequent items
print(freq_items(data))

# bar plot for categorical features
bar_plot(data,  display=True)

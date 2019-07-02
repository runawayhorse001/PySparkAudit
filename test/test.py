# set up the Spark Session
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PySpark Data Audit example")\
                    .config("spark.some.config.option", "some-value").getOrCreate()

# import PySparkAudit function
from PySparkAudit import trend_plot, auditing

# load data
data = spark.read.format('com.databricks.spark.csv')\
            .options(header='true',inferschema='true').load("Heart.csv",header=True);

# set output path
path = '/home/feng/Desktop/unit_test/test'


# auditing in oue function 
auditing(data, output_dir=path, types='day', tracking=False)
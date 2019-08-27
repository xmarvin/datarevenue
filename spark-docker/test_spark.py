#import pandas as pd

from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pyspark as spark
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark import SQLContext
import os

os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"]="python3.6"


conf = SparkConf().setAppName('app').setMaster('spark://spark-master:7077')
sc = SparkContext(conf=conf)
hc = SparkSession(sc)


def process_df(df):
  def points2group(score):
    splits = [84, 89, 95]
    for (index, split) in enumerate(splits):
      if score < split:
        return index
    return len(splits)

  df['group'] = df['points'].apply(points2group)
  df['country'] = df['country'].fillna('unk')
  df['price'] = df['price'].fillna(df['price'].mean())
  return df[['description', 'price', 'group']]

def to_spark_df(fin):
  df = pd.read_csv(fin)
  df = process_df(df)
  df = hc.createDataFrame(df)
  return df


sqlContext = SQLContext(sc)

#train = to_spark_df("/usr/share/data/raw/wine_dataset.csv")
train = sqlContext.read.csv("/usr/share/data/raw/wine_dataset.csv")
print(train.show(3))


# tokenizer = Tokenizer(inputCol="description", outputCol="words")
# wordsData = tokenizer.transform(train)
#
# hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
# tf = hashingTF.transform(wordsData)
#
# idf = IDF(inputCol="rawFeatures", outputCol="features")
# idfModel = idf.fit(tf)
# idfModel.save('/usr/share/data/train_model/idf.model')
# tfidf = idfModel.transform(tf)
#
# lr = LogisticRegression(featuresCol="features", labelCol='group', regParam=0.1)
# lrModel = lr.fit(tfidf)
# res_train = lrModel.transform(tfidf)
# res_train.show(5)


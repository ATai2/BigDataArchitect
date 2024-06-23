package com.atai.ml

import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession


object LinearRegression002 {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setMaster("local")
    val spark = SparkSession.builder().config(conf)
      .appName("LinearRegression")
      .getOrCreate()

    var data = spark.read.format("libsvm")
      .load("data/mllib/sample_linear_regression_data.txt")

    println("==========================")
    data.show(10, false)
    import spark.implicits._
    data = data.rdd.map(row => {
      val features = row.getAs[SparseVector]("features")
      val label = row.getAs[Double]("label")
      val featureArr = features.toDense.toArray
      (label, new DenseVector(featureArr.+:(featureArr(0))))
    }).toDF("label", "features")

//    data = data.rdd.map(row => {
//      val features = row.getAs[SparseVector]("features")
//      val label = row.getAs[Double]("label")
//      val featureArr = features.toDense.toArray
//      (label, new DenseVector(featureArr.+:(1.0)))
//    }).toDF("label", "features")

    data.show(10, false)

    val DFS = data.randomSplit(Array(0.8, 0.2), 1)
    val (training, test) = (DFS(0), DFS(1))

    val lr = new LinearRegression()
      .setMaxIter(10)
    //      .setTol()
    val model = lr.fit(training)
    println(s"Coefficients:${model.coefficients} Intercept: ${model.intercept}")





  }
}

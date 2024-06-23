package com.atai.ml

import org.apache.spark.mllib.linalg.SparseVector

object VectorTest {
  def main(args: Array[String]): Unit = {
    val vector = new SparseVector(10, Array(1, 3, 5), Array(2.0, 3.0, 4.0))
    println(vector.toDense)
  }

}

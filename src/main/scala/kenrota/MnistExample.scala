package kenrota

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.factory.Nd4j
import scala.io

object MnistExample extends App {
  val numInputNodes = 784
  val numHiddenNodes = 100
  val numOutputNodes = 10
  val learningLate = 0.1
  val numMnistTrain = 60000
  val numMnistTest = 10000
  val epoch = 5
  val nn = new NeuralNetwork(numInputNodes, numHiddenNodes, numOutputNodes, learningLate)

  def loadMnist(filename: String, numRows: Int): Array[Array[Double]] = {
    val numCols = 1 + numInputNodes
    val rows = Array.ofDim[Double](numRows, numCols)
    val bufferedSource = io.Source.fromFile(filename)
    var count = 0
    for (line <- bufferedSource.getLines) {
      rows(count) = line.split(",").map(_.trim.toDouble)
      count += 1
    }
    bufferedSource.close
    rows
  }

  val trainDataSet = loadMnist("src/main/resources/mnist_train.csv", numMnistTrain)

  for (_ <- 1 to epoch) {
    for (trainData <- trainDataSet) {
      val inputs: INDArray = Nd4j.create(trainData).get(NDArrayIndex.interval(1, trainData.size)).div(255).mul(0.99).add(0.01)
      var targets: INDArray = Nd4j.zeros(numOutputNodes).add(0.01)
      targets.putScalar(trainData(0).toInt, 0.99)
      nn.train(inputs, targets)
    }
  }

  val testDataSet = loadMnist("src/main/resources/mnist_test.csv", numMnistTest)
  var numCorrects = 0

  for (testData <- testDataSet) {
    val correctLabel = testData(0).toInt

    val inputs: INDArray = Nd4j.create(testData).get(NDArrayIndex.interval(1, testData.size)).div(255).mul(0.99).add(0.01)

    val outputs: INDArray = nn.predict(inputs)
    val label = Nd4j.argMax(outputs).aminNumber().intValue()

    if (correctLabel == label) {
      numCorrects += 1
    }
  }

  println(s"Performance: ${numCorrects.toDouble / testDataSet.size}")
}

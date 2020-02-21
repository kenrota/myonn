package kenrota

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution

class NeuralNetwork(
  val numInputNodes: Int,
  val numHiddenNodes: Int,
  val numOutputNodes: Int,
  val learningRate: Double) {

  var wIH: INDArray = Nd4j.rand(numHiddenNodes, numInputNodes).sub(0.5)
  var wHO: INDArray = Nd4j.rand(numOutputNodes, numHiddenNodes).sub(0.5)

  //var wIH: INDArray = Nd4j.rand(Array(numHiddenNodes, numInputNodes), new NormalDistribution(0, Math.pow(numHiddenNodes, -0.5)))
  //var wHO: INDArray = Nd4j.rand(Array(numOutputNodes, numHiddenNodes), new NormalDistribution(0, Math.pow(numOutputNodes, -0.5)))

  def train(inputs: INDArray, targets: INDArray): Unit = {
    val hiddenInputs: INDArray = wIH.mmul(inputs.transpose())
    val hiddenOutputs: INDArray = Transforms.sigmoid(hiddenInputs)

    val finalInputs: INDArray = wHO.mmul(hiddenOutputs)
    val finalOutputs: INDArray = Transforms.sigmoid(finalInputs)

    val outputErrors: INDArray = targets.transpose().sub(finalOutputs)
    val hiddenErrors: INDArray = wHO.transpose().mmul(outputErrors)

    wHO = wHO.add((outputErrors.mul(finalOutputs)).mul(finalOutputs.rsub(1.0)).mmul(hiddenOutputs.transpose()).mul(learningRate))
    wIH = wIH.add((hiddenErrors.mul(hiddenOutputs)).mul(hiddenOutputs.rsub(1.0)).mmul(inputs).mul(learningRate))

    ()
  }

  def predict(inputs: INDArray): INDArray = {
    val hiddenInputs: INDArray = wIH.mmul(inputs.transpose())
    val hiddenOutputs: INDArray = Transforms.sigmoid(hiddenInputs)

    val finalInputs: INDArray = wHO.mmul(hiddenOutputs)
    val finalOutputs: INDArray = Transforms.sigmoid(finalInputs)

    finalOutputs
  }
}

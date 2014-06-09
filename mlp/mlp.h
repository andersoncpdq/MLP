/*
 * mlp.h
 *
 *  Created on: 20/05/2014
 *      Author: anderson
 */

#ifndef MLP_H_
#define MLP_H_

#include "layer.h"

class MLP
{
public:

	std::vector<Layer> layers;
	std::vector<double> outputError;
	std::vector<double> averageErrorTrain;
	std::vector<double> averageErrorTest;
	std::vector< std::vector<int> > confusionMatrix;

	int numEpochs;
	int numLayers;
	int numOutputs;
	double learningRate;
	double momentum;
	double acceptableError;
	int lastSeason;
	double finalError;

	MLP();
	void config(int numLayers, int numClasses);
	void layerConfig(int layerNumber, int numNeurons, functionType activateFunction);
	void trainingConfig(int numEpochs, double acceptableError, double learningRate, double momentum);
	void training(std::vector< std::vector<double> >& dataTrain, std::vector< std::vector<double> >& targetTrain,
				  std::vector< std::vector<double> >& dataTest, std::vector< std::vector<double> >& targetTest);
	void forward(std::vector<double>& input);
	void backward(std::vector<double>& input);
	//void getAverageSquareError(int epoch, int example, int dataSize, std::vector<double>& target);
	void getErrorValues(std::vector<double>& target, int epoch, int flag);
	//bool stoppedTest(int epoch);
	bool stoppedTest(int epoch);
	double activationFunction(double pot, functionType function);
	double derivativeActivationFunction(double pot, functionType function);
	double sumDeltaTimesWeights(int layerNumber, int neuronNumber);

	void printWeights();
	void printConfusionMatrix();
};

#endif /* MLP_H_ */

/*
 * mlp.cpp
 *
 *  Created on: 20/05/2014
 *      Author: anderson
 */

#include "mlp.h"
using namespace std;

MLP::MLP()
{
	numLayers 		= 0;
	numEpochs 		= 0;
	numOutputs 		= 0;
	acceptableError = 0.0;
	learningRate 	= 0.0;
	momentum 		= 0.0;
	lastSeason 		= 0;
	finalError 		= 0.0;
}

void MLP::config(int numLayers, int numClasses)
{
	this->numLayers = numLayers;
	layers.resize(this->numLayers);

	confusionMatrix.resize(numClasses);
	for(int i = 0; i < numClasses; i++)
		confusionMatrix[i].resize(numClasses);
}

void MLP::layerConfig(int layerNumber, int numNeurons, functionType activateFunction)
{
	if(layerNumber == 0)
		layers[layerNumber].init(numNeurons, 0, sigmoid); // a camada de entrada nao possui sinapses.
	else
		layers[layerNumber].init(numNeurons, layers[layerNumber - 1].numNeurons, activateFunction);

	if(layerNumber == (numLayers - 1))
	{
		numOutputs = layers[layerNumber].numNeurons;
		outputError.resize(numOutputs);
	}
}

void MLP::trainingConfig(int numEpochs, double acceptableError, double learningRate, double momentum)
{
	this->numEpochs 		= numEpochs;
	this->acceptableError 	= acceptableError;
	this->learningRate 		= learningRate;
	this->momentum 			= momentum;

	averageErrorTrain.resize(this->numEpochs);
	averageErrorTest.resize(this->numEpochs);
}

void MLP::training(std::vector< std::vector<double> >& dataTrain, std::vector< std::vector<double> >& targetTrain,
				   std::vector< std::vector<double> >& dataTest, std::vector< std::vector<double> >& targetTest)
{
	int flag;
	int realClass;
	int predClass;
	double max;

	for(int epoch = 0; epoch < numEpochs; epoch++)
	{
		// Treinamento
		flag = 0;
		for(unsigned int example = 0; example < dataTrain.size(); example++)
		{
			forward(dataTrain[example]);
			getErrorValues(targetTrain[example], epoch, flag);
			backward(dataTrain[example]);
		}

		averageErrorTrain[epoch] /= dataTrain.size();

		// Validacao
		flag = 1;
		realClass = 0;
		for(unsigned int ex = 0; ex < dataTest.size(); ex++)
		{
			forward(dataTest[ex]);
			getErrorValues(targetTest[ex], epoch, flag);

			// Preencher a Matriz de Confusao somente na ultima epoca de teste
			if( averageErrorTrain[epoch] < acceptableError || epoch == (numEpochs - 1))
			{
				for(int j = 0; j < numOutputs; j++)
				{
					if(targetTest[ex][j] == 1)
						realClass = j;
				}

				max = layers[numLayers - 1].output[0];
				predClass = 0;
				double aux = 0.0;

				for(int i = 1; i < numOutputs; i++)
				{
					aux = layers[numLayers - 1].output[i];
					if(max < aux)
					{
						max = aux;
						predClass = i;
					}
				}
				confusionMatrix[realClass][predClass]++;
			}
		}

		averageErrorTest[epoch] /= dataTest.size();

		if( stoppedTest(epoch) == true )
			break;
	}
}

/*bool MLP::stoppedTest(int epoch)
{
	bool flag = false;

	lastSeason = epoch;
	finalError = averageSquareError[epoch];

	cout << "Epoca: " << epoch << " :: Erro: " << finalError << endl;

	if(finalError < acceptableError)
	{
		flag = true;
		cout << "Erro Final = " << finalError << " < " << acceptableError << endl;
	}

	return flag;
}*/

bool MLP::stoppedTest(int epoch)
{
    bool flag = false;

    lastSeason = epoch;
    finalError = averageErrorTrain[epoch];

    cout << "Epoch: " << epoch << " :: Erro: " << averageErrorTrain[epoch] << endl;

    if( averageErrorTrain[epoch] < acceptableError )
    {
        flag = true;
        cout << "Erro Medio Minino = " << averageErrorTrain[epoch] << " < " << acceptableError << endl;
    }
    return flag;
}

void MLP::forward(std::vector<double>& input)
{
	for(int l = 1; l < numLayers; l++)
	{
		for(int n = 0; n < layers[l].numNeurons; n++)
		{
			if(l == 1)
				layers[l].potential[n] = layers[l].neurons[n].activationPotential(input);
			else
				layers[l].potential[n] = layers[l].neurons[n].activationPotential(layers[l-1].output);

			layers[l].output[n] = activationFunction(layers[l].potential[n], layers[l].activationFunction);
		}
	}
}

void MLP::backward(std::vector<double>& input)
{
	// Obter deltas
	for(int l = (numLayers - 1); l >= 1; l--)
	{
		for(int n = 0; n < layers[l].numNeurons; n++)
		{
			if(l == (numLayers - 1))
				layers[l].delta[n] = outputError[n] * derivativeActivationFunction(layers[l].output[n], layers[l].activationFunction);
			else
				layers[l].delta[n] = sumDeltaTimesWeights(l+1, n) * derivativeActivationFunction(layers[l].output[n], layers[l].activationFunction);
		}
	}

	// Ajuste dos pesos sinapticos
	double inputLayer;
	double temp;

	for(int l = (numLayers - 1); l >= 1; l--)
	{
		for(int n = 0; n < layers[l].numNeurons; n++)
		{
			for(int w = 0; w < layers[l].numSynapsesPerNeuron; w++)
			{
				if(l == 1)
					inputLayer = input[w];
				else
					inputLayer = layers[l-1].output[w];

				temp = layers[l].neurons[n].weights[w];

				layers[l].neurons[n].weights[w] += momentum * (layers[l].neurons[n].weights[w] - layers[l].neurons[n].weightsOld[w])
												   + learningRate * layers[l].delta[n] * inputLayer;

				layers[l].neurons[n].weightsOld[w] = temp;
			}

            temp = layers[l].neurons[n].bias;

            layers[l].neurons[n].bias += momentum * (layers[l].neurons[n].bias - layers[l].neurons[n].biasOld)
            							 + learningRate * layers[l].delta[n];

            layers[l].neurons[n].biasOld = temp;
		}
	}
}

/*void MLP::getAverageSquareError(int epoch, int example, int dataSize, std::vector<double>& target)
{
	std::vector<double> squareError;
	squareError.resize(dataSize);

	double sumSquareError = 0.0;

	for(int i = 0; i < numOutputs; i++)
	{
		outputError[i] = target[i] - layers[numLayers - 1].output[i];
		sumSquareError += pow(outputError[i], 2);
	}

	squareError[example] = 0.5 * sumSquareError;

	if( example == (dataSize - 1) )
	{
		sumSquareError = 0.0;
		for(int e = 0; e < dataSize; e++)
			sumSquareError += squareError[e];

		averageSquareError[epoch] = sumSquareError / dataSize;
	}
}*/

void MLP::getErrorValues(std::vector<double>& target, int epoch, int flag)
{
    double energy = 0;

    for(int i = 0; i < numOutputs; i++)
    {
        outputError[i] = target[i] - layers[numLayers - 1].output[i];
        energy += pow(outputError[i], 2);
    }

    if(flag == 0)
    	averageErrorTrain[epoch] += 0.5 * energy;
    else
    	averageErrorTest[epoch] += 0.5 * energy;
}

double MLP::activationFunction(double pot, functionType function)
{
	double result = 0.0;

	switch(function)
	{
		case sigmoid:
			result = 1.0 / (1.0 + exp(-pot));
			break;
		default:
			cout << "Funcao de ativacao incorreta!" << endl;
			break;
	}

	return result;
}

double MLP::derivativeActivationFunction(double pot, functionType function)
{
	double result = 0.0;

	switch(function)
	{
		case sigmoid:
			result = pot * (1.0 - pot);
			break;
		default:
			cout << "Funcao de ativacao derivada incorreta!" << endl;
			break;
	}

	return result;
}

double MLP::sumDeltaTimesWeights(int layerNumber, int neuronNumber)
{
	double result = 0.0;

	for(int n = 0; n < layers[layerNumber].numNeurons; n++)
		result += layers[layerNumber].delta[n] * layers[layerNumber].neurons[n].weights[neuronNumber];

	return result;
}

void MLP::printWeights()
{
	for(unsigned int l = 0; l < layers.size(); l++)
	{
		cout << "Layer " << l << ":" << endl;
		for(int n = 0; n < layers[l].numNeurons; n++)
		{
			cout << "	Neuron " << n << ":" << endl;
			for(int w = 0; w < layers[l].neurons[n].numSynapses; w++)
				cout << "		Weight " << w << " = " << layers[l].neurons[n].weights[w] << endl;
		}
	}
}

void MLP::printConfusionMatrix()
{
	cout << endl << "##### Matriz de Confusao #####" << endl << endl;
	cout << "Real Predita" << endl;
	for(unsigned int i = 0; i < confusionMatrix.size(); i++)
	{
		for(unsigned int j = 0; j < confusionMatrix[i].size(); j++)
			cout << " [" << i << "]   [" << j << "]  =  " << confusionMatrix[i][j] << endl;
		cout << endl;
	}
}

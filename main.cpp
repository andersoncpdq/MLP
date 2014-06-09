/*
 * main.cpp
 *
 *  Created on: 20/05/2014
 *      Author: anderson
 */

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <iomanip>

#include "mlp/mlp.h"

using namespace std;

void fileInput(const char dataset[], std::vector< std::vector<double> >& data, std::vector< std::vector<double> >& target);
void fileOutputError(const char dataset[], std::vector<double>);
void fileOutputWeights(const char dataset[], std::vector<Layer>);
void fileOutputConfusionMatrix(const char dataset[], std::vector< std::vector<int> >);

int main()
{
	int numEpochs 			= 3000;
	double acceptableError 	= 0.0001;
	double learningRate 	= 0.2;
	double momentum 		= 0.3;

	std::vector< std::vector<double> > trainingData;
	std::vector< std::vector<double> > trainingTarget;

	std::vector< std::vector<double> > validationData;
	std::vector< std::vector<double> > validationTarget;

    fileInput("datasets/cefaleia_lda.data", trainingData, trainingTarget);
    fileInput("datasets/cefaleia_lda.test", validationData, validationTarget);

    cout << "Multilayer Perceptron" << endl << endl;
    cout << " A G U A R D E   O   P R O C E S S A M E N T O" << endl;

    int numInputs = trainingData[0].size();
    int numOutputs = trainingTarget[0].size();

	MLP mlp;

	// Configuracao
	mlp.config(3, numOutputs);
	mlp.layerConfig(0, numInputs, sigmoid); 	// camada de entrada
	mlp.layerConfig(1, 8, sigmoid);
	mlp.layerConfig(2, numOutputs, sigmoid);	// camada de saida

	// Treinamento
	mlp.trainingConfig(numEpochs, acceptableError, learningRate, momentum);
	mlp.training(trainingData, trainingTarget, validationData, validationTarget);

	mlp.printConfusionMatrix();

	fileOutputError("cefaleia_errorTrain.txt", mlp.averageErrorTrain);
	fileOutputError("cefaleia_errorTest.txt", mlp.averageErrorTest);
	fileOutputWeights("cefaleia_weights.txt", mlp.layers);
	fileOutputConfusionMatrix("cefaleia_confusionMatrix.txt", mlp.confusionMatrix);

	return 0;
}

void fileInput(const char dataset[], std::vector< std::vector<double> >& data, std::vector< std::vector<double> >& target)
{
	int numExamples;
	int numInputs;
	int numTargets;
	const int TITLE_LENGHT = 100;
	char title[TITLE_LENGHT];

    // abertura do arquivo atraves do construtor ifstream.
    ifstream inFile(dataset, ios::in);

    // termina o programa caso o arquivo nao possa ser aberto.
    if( !inFile )
    {
        cerr << "Um arquivo nao pode ser aberto!" << endl;
        exit(1);
    }

    inFile.getline(title, TITLE_LENGHT, '\n');

    inFile >> numExamples;
    inFile >> numInputs;
    inFile >> numTargets;

    inFile.getline(title, TITLE_LENGHT, '\n');
    inFile.getline(title, TITLE_LENGHT, '\n');

    data.resize(numExamples);
    target.resize(numExamples);

    for(int i = 0; i < numExamples; i++)
    {
        data[i].resize(numInputs);
        target[i].resize(numTargets);
    }

    for(unsigned int i = 0; i < data.size(); i++)
    {
    	//cout << i << ": ";
        for (unsigned int j = 0; j < data[i].size(); j++)
        {
            inFile >> data[i][j];
            //cout << data[i][j] << " ";
        }

        //cout << "output: ";
        for (unsigned int j = 0; j < target[i].size(); j++)
        {
            inFile >> target[i][j];
            //cout << target[i][j] << " ";
        }
        //cout << endl;
    }

    inFile.close();
}

void fileOutputError(const char dataset[], std::vector<double> averageError)
{
    ofstream outFile(dataset, ios::out);

    if (!outFile)
    {
    	cerr << "O log de erro medio não pode ser criado!" << endl;
    	exit(1);
    }

    for(unsigned int i = 0; i < averageError.size(); i++)
    	outFile << fixed << showpoint << setprecision( 20 ) << averageError[i] << endl;
}

void fileOutputWeights(const char dataset[], std::vector<Layer> layers)
{
    ofstream outFile(dataset, ios::out);

    if (!outFile)
    {
    	cerr << "O log dos pesos não pode ser criado!" << endl;
    	exit(1);
    }

	for(unsigned int l = 1; l < layers.size(); l++)
	{
		for(int n = 0; n < layers[l].numNeurons; n++)
		{
			for(int w = 0; w < layers[l].neurons[n].numSynapses; w++)
				outFile << n << "\t" << layers[l].neurons[n].weights[w] << "\n";
		}
	}
}

void fileOutputConfusionMatrix(const char dataset[], std::vector< std::vector<int> > confusionMatrix)
{
    ofstream outFile(dataset, ios::out);

    if (!outFile)
    {
    	cerr << "O log da matriz de confusao não pode ser criado!" << endl;
    	exit(1);
    }

    outFile << "##### Matriz de Confusao #####" << "\n\n";
    outFile << "Real Predita" << "\n";
	for(unsigned int i = 0; i < confusionMatrix.size(); i++)
	{
		for(unsigned int j = 0; j < confusionMatrix[i].size(); j++)
			outFile << " [" << i << "]   [" << j << "]  =  " << confusionMatrix[i][j] << "\n";
		outFile << "\n";
	}
}

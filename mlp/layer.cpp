/*
 * layer.cpp
 *
 *  Created on: 20/05/2014
 *      Author: anderson
 */

#include "layer.h"

Layer::Layer()
{
	numNeurons 			 = 0;
	numSynapsesPerNeuron = 0;
	activationFunction 	 = sigmoid;
}

void Layer::init(int numNeurons, int numSynapsesPerNeuron, functionType activationFunction)
{
	this->numNeurons 			= numNeurons;
	this->numSynapsesPerNeuron 	= numSynapsesPerNeuron;
	this->activationFunction 	= activationFunction;

	neurons.resize(this->numNeurons);
	potential.resize(this->numNeurons);
	output.resize(this->numNeurons);
	delta.resize(this->numNeurons);

	for(int i = 0; i < numNeurons; i++)
		neurons[i].init(this->numSynapsesPerNeuron);
}

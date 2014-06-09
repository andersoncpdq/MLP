/*
 * neuron.cpp
 *
 *  Created on: 20/05/2014
 *      Author: anderson
 */

#include "neuron.h"

Neuron::Neuron()
{
	bias 		= 0.0;
	biasOld 	= 0.0;
	numSynapses = 0;
}

void Neuron::init(int numSynapses)
{
	this->numSynapses = numSynapses;
	weights.resize(this->numSynapses);
	weightsOld.resize(this->numSynapses);
	weightsInit();
}

void Neuron::weightsInit()
{
	// seed
	mt_seed();

	for(int i = 0; i < numSynapses; i++)
	{
		weights[i]		= mt_ldrand(); // gera numeros pseudo-aleatorios entre 0.0 - 1.0
		weightsOld[i] 	= weights[i];
	}

	bias 	= mt_ldrand();
	biasOld = bias;
}

double Neuron::activationPotential(std::vector<double>& input)
{
	double acc = bias;

	for(int i = 0; i < numSynapses; i++)
		acc += weights[i] * input[i];

	return acc;
}

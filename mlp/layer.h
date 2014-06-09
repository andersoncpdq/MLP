/*
 * layer.h
 *
 *  Created on: 20/05/2014
 *      Author: anderson
 */

#ifndef LAYER_H_
#define LAYER_H_

#include "neuron.h"

enum functionType
{
	sigmoid
};

class Layer
{
public:
	std::vector<Neuron> neurons;
	std::vector<double> potential;
	std::vector<double> output;
	std::vector<double> delta;
	int numNeurons;
	int numSynapsesPerNeuron;
	functionType activationFunction;

	Layer();
	void init(int numNeurons, int numSynapsesPerNeuron, functionType activationFunction);
};

#endif /* LAYER_H_ */

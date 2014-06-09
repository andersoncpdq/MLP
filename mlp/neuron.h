/*
 * neuron.h
 *
 *  Created on: 20/05/2014
 *      Author: anderson
 */

#ifndef NEURON_H_
#define NEURON_H_

#include <cmath>
#include <vector>
#include "/home/anderson/workspace/MLP/mersenne/mtwist.h"

class Neuron
{
public:
	std::vector<double> weights;
	std::vector<double> weightsOld;
	double bias;
	double biasOld;
	int numSynapses;

	Neuron();
	void init(int numSynapses);
	void weightsInit();
	double activationPotential(std::vector<double>& input);
};

#endif /* NEURON_H_ */

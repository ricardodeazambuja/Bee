# BEE - The Spiking Reservoir (LSM) Simulator

The **BEE simulator** is an open source [Spiking Neural Network (SNN)](https://en.wikipedia.org/wiki/Spiking_neural_network) simulator, [freely available](https://github.com/ricardodeazambuja/BEE/blob/master/LICENSE.md), specialised in [Liquid State Machine (LSM)](https://en.wikipedia.org/wiki/Liquid_state_machine) systems with its core functions fully implemented in C. 

![Block diagram of an LSM](https://github.com/ricardodeazambuja/BEE/blob/master/imgs/lsm.png)

It was developed together with my PhD thesis [(you can see where it was used in my publications)](http://ricardodeazambuja.com/publications/) exclusively to solve the specific problems presented by neurorobotics experiments.  

BEE uses the C library [pthreads (POSIX threads)](https://en.wikipedia.org/wiki/POSIX_Threads) in order to speed up the simulation of LSMs by processing input and output spikes in a parallel way. A Python wrapper is supplied to simplify the user interaction with the software.  
The neuron model, a special type of [(Leaky_integrate-and-fire)](https://en.wikipedia.org/wiki/Biological_neuron_model#Leaky_integrate-and-fire) with extra exponential synapses - see [1] for details, is hardcoded (fixed), following what is presented bellow, and the solution for the differential equations is calculated by the [Euler's method](https://en.wikipedia.org/wiki/Euler_method) according to the simulation's time step specified by the user. 

![Leaky Integrate and Fire with Exponential Synapses](https://github.com/ricardodeazambuja/BEE/blob/master/imgs/neuron_model.png)

The simulator has the ability to automatically generate the reservoir (liquid) in a probabilistic way (see [1] for details) according to the equation:

![Probability of connections](https://github.com/ricardodeazambuja/BEE/blob/master/imgs/conn_model.png)

An example of generated connections (red dots/lines are excitatory and blue ones inhibitory) is presented below:
![liquid example](https://github.com/ricardodeazambuja/BEE/blob/master/imgs/liquid.png)

All the parameters for the neuron model or the internal connections can be defined by the user. Also, motivated by the results presented in [Short-term plasticity in a liquid state machine biomimetic robot arm controller](https://github.com/ricardodeazambuja/IJCNN2017-2), [Short Term Plasticity (STP)](http://www.scholarpedia.org/article/Short-term_synaptic_plasticity) and time delays were not implemented in order to simplify and optimise the simulator. In its current version, it supports, at least, Linux and OS X (it was never tested by the author on any version of Windows).

## If you want to find out more details about the main simulator I'd recommend to have a look here:  
https://github.com/ricardodeazambuja/BEE/blob/master/BEE/BEE_Simulator_DEVELOPMENT.ipynb

## All the necessary files can be found here:  
https://github.com/ricardodeazambuja/BEE/tree/master/BEE


## Here is a list of published papers that use BEE (they have plenty of code examples to follow):
- [Graceful Degradation under Noise on Brain Inspired Robot Controllers](https://github.com/ricardodeazambuja/ICONIP2016)
- [Diverse, Noisy and Parallel: a New Spiking Neural Network Approach for Humanoid Robot Control](https://github.com/ricardodeazambuja/IJCNN2016)
- [Short-Term Plasticity in a Liquid State Machine Biomimetic Robot Arm Controller](https://github.com/ricardodeazambuja/IJCNN2017)
- [Neurorobotic Simulations on the Degradation of Multiple Column Liquid State Machines](https://github.com/ricardodeazambuja/IJCNN2017-2)
- [Sensor Fusion Approach Using Liquid StateMachines for Positioning Control](https://github.com/ricardodeazambuja/I2MTC2017-LSMFusion)

## If you are using BEE in your work, please, send me the link and I will add it here :)

## References
1. Maass, Wolfgang, Thomas Natschläger, and Henry Markram. “Real-Time Computing without Stable States: A New Framework for Neural Computation Based on Perturbations.” Neural Computation 14, no. 11 (November 2002): 2531–60.

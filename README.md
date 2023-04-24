# Bee - The Spiking Reservoir (LSM) Simulator

The **Bee simulator** is an open source [Spiking Neural Network (SNN)](https://en.wikipedia.org/wiki/Spiking_neural_network) simulator, [freely available](https://github.com/ricardodeazambuja/BEE/blob/master/LICENSE.md), specialised in [Liquid State Machine (LSM)](https://en.wikipedia.org/wiki/Liquid_state_machine) systems with its core functions fully implemented in C. 

![Block diagram of an LSM](https://github.com/ricardodeazambuja/BEE/blob/master/imgs/lsm.png)

It was developed together with my PhD thesis [(you can see where it was used in my publications)](http://ricardodeazambuja.com/publications/) exclusively to solve the specific problems presented by neurorobotics experiments.  

Bee uses the C library [pthreads (POSIX threads)](https://en.wikipedia.org/wiki/POSIX_Threads) in order to speed up the simulation of LSMs by processing input and output spikes in a parallel way. A Python wrapper is supplied to simplify the user interaction with the software.  
The neuron model, a special type of [(Leaky_integrate-and-fire)](https://en.wikipedia.org/wiki/Biological_neuron_model#Leaky_integrate-and-fire) with extra exponential synapses - see [1] for details, is hardcoded (fixed), following what is presented bellow, and the solution for the differential equations is calculated by the [Euler's method](https://en.wikipedia.org/wiki/Euler_method) according to the simulation's time step specified by the user. 

![Leaky Integrate and Fire with Exponential Synapses](https://github.com/ricardodeazambuja/BEE/blob/master/imgs/neuron_model.png)

The simulator has the ability to automatically generate the reservoir (liquid) in a probabilistic way (see [1] for details) according to the equation:

![Probability of connections](https://github.com/ricardodeazambuja/BEE/blob/master/imgs/conn_model.png)

An example of generated connections (red dots/lines are excitatory and blue ones inhibitory) is presented below:
![liquid example](https://github.com/ricardodeazambuja/BEE/blob/master/imgs/liquid.png)     
**Note (24/04/2023): I just realised, by looking at the image above, that it could be a better idea to enforce neurons to have connections to two other neurons or input/output, input/neuron or output/neuron connections, otherwise that neuron won't contribute to the internal dynamics and may even waste an input or output that falls on it. In addition to this minimum amount of "neighbour" connections, it should have a guarantee that every neuron has at least a path to one of the liquid's input and a path to one output.**

All the parameters for the neuron model or the internal connections can be defined by the user. Also, motivated by the results presented in [Short-term plasticity in a liquid state machine biomimetic robot arm controller](https://github.com/ricardodeazambuja/IJCNN2017-2), [Short Term Plasticity (STP)](http://www.scholarpedia.org/article/Short-term_synaptic_plasticity) and time delays were not implemented in order to simplify and optimise the simulator. In its current version, it supports, at least, Linux and OS X (it was never tested by the author on any version of Windows).

## If you want to find out more details about the main simulator I'd recommend to have a look here:  
https://github.com/ricardodeazambuja/BEE/blob/master/BEE/BEE_Simulator_DEVELOPMENT.ipynb

## All the necessary files can be found here:  
https://github.com/ricardodeazambuja/BEE/tree/master/BEE


## Here is a list of published papers that use Bee (they have plenty of code examples to follow):
- [Graceful Degradation under Noise on Brain Inspired Robot Controllers](https://github.com/ricardodeazambuja/ICONIP2016)
- [Diverse, Noisy and Parallel: a New Spiking Neural Network Approach for Humanoid Robot Control](https://github.com/ricardodeazambuja/IJCNN2016)
- [Short-Term Plasticity in a Liquid State Machine Biomimetic Robot Arm Controller](https://github.com/ricardodeazambuja/IJCNN2017)
- [Neurorobotic Simulations on the Degradation of Multiple Column Liquid State Machines](https://github.com/ricardodeazambuja/IJCNN2017-2)
- [Sensor Fusion Approach Using Liquid StateMachines for Positioning Control](https://github.com/ricardodeazambuja/I2MTC2017-LSMFusion)

## If you are using Bee in your work, please, send me the link and I will add it here :)

## Ideas for new projects derived from my work
- [Here you can find some of the ideas that I never had time to publish](https://github.com/ricardodeazambuja/SNN-Experiments)

## Videos
- [Animation of what happens inside a LSM](https://www.youtube.com/watch?v=_xm77cxpXV8)
- [Controlling the BAXTER robot using Liquid State Machines](https://www.youtube.com/watch?v=4gF7mfjGllA)

## References
1. Maass, Wolfgang, Thomas Natschläger, and Henry Markram. “Real-Time Computing without Stable States: A New Framework for Neural Computation Based on Perturbations.” Neural Computation 14, no. 11 (November 2002): 2531–60.  


## Other projects you may like to check:
* [colab_utils](https://github.com/ricardodeazambuja/colab_utils/): Some useful (or not so much) Python stuff for Google Colab notebooks
* [ExecThatCell](https://github.com/ricardodeazambuja/ExecThatCell): (Re)Execute a Jupyter (colab) notebook cell programmatically by searching for its label.
* [Maple-Syrup-Pi-Camera](https://github.com/ricardodeazambuja/Maple-Syrup-Pi-Camera): Low power('ish) AIoT smart camera (3D printed) based on the Raspberry Pi Zero W and Google Coral EdgeTPU
* [The CogniFly Project](https://github.com/thecognifly): Open-source autonomous flying robots robust to collisions and smart enough to do something interesting!

http://ricardodeazambuja.com/

/*

THIS VERSION IS USING FLOAT VALUES INSTEAD OF DOUBLES BECAUSE I WAS HOPING GCC COULD DO BETTER OPTIMIZATIONS THIS WAY (VECTORIZATION)!
ANOTHER REASON TO USE FLOAT IS TO START THINKING ABOUT THE TRANSLATION OF THE CODE TO CUDA OR OPENCL...

IF DOUBLE FITS YOU BETTER, JUST SEARCH ALL OCCURENCES OF "float" AND SUBSTITUTE BY "double".


---------------------------------
BEE - The Spiking Liquid Simulator
----------------------------------
https://github.com/ricardodeazambuja


This is a VERY VERY (VERY) simple simulator. It implements the neuron model:
    dv/dt  = (ie + ii + i_offset + i_inj)/cm + (v_rest-v)/taum  [in V]
    die/dt = -ie/taue   [in nA]
    dii/dt = -ii/taui   [in nA]

    => ie is the state variable corresponding to the excitatory neurons current
    => ii is the state variable corresponding to the inhibitory neurons current
    => v is the state variable corresponding to the membrane voltage
    => cm is the membrane capacitance
    => v_rest is the membrane voltage resting potential
    => taum is the membrane time constant
    => taue is the excitatory current time constant
    => taui is the inhibitory current time constant
    => i_offset is a constant random value generated at the liquid's initialization
    => i_inj is a random value generated at each time step.

    The differential equation is solved by a simple numerical method (midpoint rule / rectangle rule?)
    One questions is: What is better? Smaller time step and simpler integration algorithm or bigger time step and
    more advanced (precise) integration????

And uses the ideas from:
Maass, Wolfgang, Thomas Natschlager, and Henry Markram. 2002.
“Real-Time Computing without Stable States: A New Framework for Neural Computation Based on Perturbations.”
Neural Computation 14 (11): 2531–60. doi:10.1162/089976602760407955.

However, the dynamic synapses (STP) and the time delays are NOT implemented in an attempt of making the system faster
and also because I'm not sure if they are necessary.

My idea is to interface only using Python (ctypes).

************************************************************************************************************************

Compile as a shared library:
make -f Makefile


OLD INFORMATION ABOUT HOW TO DEBUG:

The "-g" flag is ONLY necessary to use if you want to debug with gdb

These flags:
-ftree-vectorizer-verbose=3 -fopt-info-missed=missed.all
save the information about vectorization (can have one of the levels 0,1,2 or 3) into the file missed.all (if they are not printed on the screen).

In gdb:
1)If the program receives command line arguments:
r arg1 arg2 arg3...
2)Creates a break line at line 444 (must be executed before run or r)
b 444
3)Steps into
step or n
4)Keep going
next or n
5)Print variables content
print or p


To run the profiler:
gcc -c ./src/BEE.c -o BEE.o -lpthread -O3 -lrt -lm -std=gnu99 -g -pg ./bin/obj/randomkit.o ./bin/obj/distributions.o

then run once the executable. After that, run (on Linux, because I could not make it work on OSX...)
gprof ./BEE.o > profile.txt
Done! Read the txt file :)

*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h> // only for the memcpy inside the main()
#include <pthread.h>

// #include <stdbool.h> // it's from C99 and enables the use of the bool type
// #include <limits.h> // has the maximum values of types: e.g UCHAR_MAX

// These are the Numpy random stuff I'm reusing
#include "randomkit.h"
#include "distributions.h"


#include <unistd.h> //getpagesize, ftruncate, open, write, etc
#include <sys/mman.h> //mmap
#include <fcntl.h> //O_RDONLY
#include <sys/types.h> //caddr_t
#include <sys/stat.h>
#include <errno.h> //I'm not using to make the errors output more verbose, but I should...


//
// Sets the right function for the timing debug stuff for GCC LINUX(__linux) and OSX (__APPLE_CC__)
// in Linux it is necessary to add gcc (linker) option (RealTime): -lrt
//
#if defined (__linux__) || defined (linux)
#include <time.h> // only for the time comparisons inside the main()
double clock_my_gettime()
{
    //  time_t tv_sec; //seconds
    //  long tv_nsec;  //NANOseconds
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return ((double)now.tv_sec*1E9 + (double)now.tv_nsec); //nanoseconds
}
#endif

#if defined (__APPLE_CC__)
#include <sys/time.h>
double clock_my_gettime()
{
    // time_t       tv_sec;    //seconds since Jan. 1, 1970
    // suseconds_t  tv_usec;   //and MICROseconds
    struct timeval now;
    int rv = gettimeofday(&now, NULL);
    if (rv) return rv; //rv=0 means OK!
    return ((double)now.tv_sec*1E9 + (double)now.tv_usec*1E3); //nanoseconds
}
#endif
//
// End of the timing functions
//



// This is used to define when realloc is going to ask for more memory
// inside the SpikingLiquid_generate_connections_less_memory_thread function:
// (I need to check which number is the optimal to increase speed)
int MAX_CONNECTIONS;

// This structure is used during the generation of the liquid's structure.
typedef struct
{
    int x;
    int y;
    int z;
}Array3D;

// Here we define the maximum number of threads
#define MAX_THREAD 50


// This should be the maximum number of connections a neuron can get
#define MAX_INDIVIDUAL_CONNECTIONS 10



// SpkLiq_liquid_paramters
// #1:  CGupta       # Parameter used at the connection probability
// #2:  UMarkram     # Use (U) - Parameter used at the Dynamic Synapse
// #3:  DMarkram     # Time constant for Depression (tau_rec) - used at the Dynamic Synapse
// #4:  FMarkram     # Time constant for Facilitation (tau_facil) - used at the Dynamic Synapse
// #5:  AMaass       # Connection gain (weight) (in nA)
// #6:  Delay_trans  # Transmission delay
//
// The Dynamic Synapses (STP - Short Term Plasticity) and the Transmission delay are not implemented... yet.
// But the time when the neuron spikes is saved inside the SpkLiq_spike_time array.
//
float SpkLiq_liquid_parameters[2][2][6] = {{
                                              { 0.1  ,  0.32 ,  0.144,  0.06 ,  -2.8  ,  0.8  },
                                              { 0.4  ,  0.25 ,  0.7  ,  0.02 ,  -3.0  ,  0.8  }
                                          },
                                          {
                                              { 0.2  ,  0.05 ,  0.125,  1.2  ,  1.6  ,  0.8  },
                                              { 0.3  ,  0.5  ,  1.1  ,  0.05 ,  1.2  ,  1.5  }
                                          }};


//
// Global variables
// These variables are global because they are used by most of the functions and would be a pain to pass as parameters!
// The first version of this algorithm was developed in an object-oriented way (Python), so these global variables were inside the main class.
// Yes, I know about global variables being bad practice, but sometimes it is faster to use them directly instead of moving memory contents, creating new pointers, etc (???)
//


// User's input variables
int SpkLiq_net_shape[3]; // the sides of the cuboid modeling the liquid's shape {x,y,z} (unit less, because it's a relative measurement)
float SpkLiq_lbd_value; // value that controls the probability of a connection inside the liquid (see Maass,Markram 2002)
float SpkLiq_lbd_value_opt; // because the variable above is used a lot during the liquid structure creation: 1/(SpkLiq_lbd_value*SpkLiq_lbd_value)
float SpkLiq_step; // simulation time step size (in seconds)
float SpkLiq_taum; // neuron membrane time constant (in seconds)
float SpkLiq_cm; // neuron membrane capacitance (in Farads)
float SpkLiq_taue; // excitatory neuron decay time constant (in seconds)
float SpkLiq_taui; // inhibitory neuron decay time constant (in seconds)
float SpkLiq_refractory[2]; // refractory periods {excitatory,inhibitory} (in seconds)
unsigned int SpkLiq_user_seeds[5]; //seeds (used in the initialization of the random states) supplied by the user
float SpkLiq_vresets[2]; // stores the reset values used with the reset value random distribution
float SpkLiq_vthres; // stores the membrane voltage value used to verify if the neuron is going to spike (in Volts)
float SpkLiq_vrest; // stores the voltage value the membrane must assume after total discharge (in Volts)
float SpkLiq_inhibitory_percentage; // the percentage of inhibitory neurons in the liquid (unit less, %)
float SpkLiq_membrane_rand[2]; //range of the possible values to the initialization of the membrane voltage (in Volts)
float SpkLiq_current_rand[2]; //range of the possible values to the initialization of the constant noisy offset current (in Amperes)
float SpkLiq_noisy_current_rand; //max range of the possible values to the initialization of the variable noisy current (in Amperes)
float SpkLiq_min_perc = 0.01; //minimum percentage based on distance, otherwise consider equal to zero.


// Random Seeds - MASTERS
unsigned int SpkLiq_rand_r_seeds[5]; //stores the random seeds variables used by the rk_seed function


// The "restrict" keyword comes from C99:
// http://en.wikipedia.org/wiki/Restrict

// These are the state variables necessary to keep the state of the simulation
float *restrict SpkLiq_neurons_membrane; //the membrane potentials for all neurons
float *restrict SpkLiq_neurons_membrane_init; //the INITIAL membrane potentials for all neurons
float *restrict SpkLiq_neurons_exc_curr; //excitatory currents levels for all neurons
float *restrict SpkLiq_neurons_inh_curr; //inhibitory currents levels for all neurons
float *restrict SpkLiq_refrac_timer; //timers used to control the refractory period
int *restrict SpkLiq_test_vthres; //controls which neuron is above the voltage threshold and should spike
long int *restrict SpkLiq_test_vthres_bits; //used to save the spikes to disk in a faster manner
float SpkLiq_current_time; //the current simulation time (in seconds)
int SpkLiq_current_step; //saves the current step (integer)
float *restrict SpkLiq_spike_time; //stores the last time each neuron spiked

float *restrict SpkLiq_vresets_values; // stores the reset values generated by the uniform distribution

float *restrict SpkLiq_noisy_offset_currents; //values of the neuron model constant noisy offset currents
float *restrict SpkLiq_noisy_currents; //values of the neuron model variable noisy currents
float *restrict SpkLiq_refrac_values; //stores the refractory periods for all neurons in the liquid
                                     //because the refractory period is different for inhibitory and excitatory neurons

int *restrict SpkLiq_inhibitory_indices; //Indices of the inhibitory neurons inside the liquid
int *restrict SpkLiq_excitatory_indices; //Indices of the excitatory neurons inside the liquid
int *restrict SpkLiq_liquid_indices_shuffled; //used to generate the random SpkLiq_inhibitory_indices/SpkLiq_excitatory_indices

int *restrict SpkLiq_neurons_connected; //signalizes that this neuron has at least ONE connection to other neuron

int *restrict SpkLiq_pre_i = NULL; //stores the indices of the inhibitory=>? pre-synaptic connections in the liquid
int *restrict SpkLiq_pos_i = NULL; //stores the indices of the inhibitory=>? pos-synaptic connections in the liquid
float *restrict SpkLiq_w_i = NULL; //stores the weights of the connections above

int *restrict SpkLiq_pre_e = NULL; //stores the indices of the inhibitory=>? pre-synaptic connections in the liquid
int *restrict SpkLiq_pos_e = NULL; //stores the indices of the inhibitory=>? pos-synaptic connections in the liquid
float *restrict SpkLiq_w_e = NULL; //stores the weights of the connections above

int SpkLiq_number_of_neurons; //total number of neurons inside the liquid, automatically calculated based on the liquid's shape

int SpkLiq_number_of_exc_neurons; //total number of excitatory neurons in the liquid
int SpkLiq_number_of_inh_neurons; //total number of inhibitory neurons in the liquid

int SpkLiq_inh_connections; //stores how many inhibitory=>? connections were created
int SpkLiq_exc_connections; //stores how many excitatory=>? connections were created


Array3D *restrict SpkLiq_3Didx; //stores the liquid's 3D structure according to the neuron index

//
// New variables used with the multithreading
//

rk_state *SpkLiq_threads_states[5]; //Pointer to store the rk_states used by the threads

unsigned int *SpkLiq_threads_seeds[5] = {NULL,NULL,NULL,NULL,NULL};
                                       // Each row match one of the Random Variables used in the simulator (Random-0...4)
                                       // and the columns are equivalent to the thread index


int SpkLiq_threads_N; // Total number of threads used

int *restrict SpkLiq_number_of_neurons_slice;  //stores the number of neurons each thread are going to process

pthread_t *SpkLiq_threads_ptrs; // Creates the pointers used with the pthread_create
                                // These pointer need to allocate SpkLiq_threads_N memory positions


int *restrict SpkLiq_thread_id; // Tells the thread function its thread index (thread_id) making easier to control which memory
                                // positions each thread is going to write to.


// SpikingLiquid_generate_connections_less_memory_thread
int **restrict SpkLiq_pre_i_thread; //stores the intermediary array of indices (generated by the threads) of the inhibitory=>? pre-synaptic connections in the liquid
int **restrict SpkLiq_pos_i_thread; //stores the intermediary array of  indices (generated by the threads) of the inhibitory=>? pos-synaptic connections in the liquid
float **restrict SpkLiq_w_i_thread; //stores the weights of the connections above

int **restrict SpkLiq_pre_e_thread; //stores the intermediary array of  indices (generated by the threads) of the excitatory=>? pre-synaptic connections in the liquid
int **restrict SpkLiq_pos_e_thread; //stores the intermediary array of  indices (generated by the threads) of the excitatory=>? pos-synaptic connections in the liquid
float **restrict SpkLiq_w_e_thread; //stores the weights of the connections above

int *restrict SpkLiq_inh_connections_thread; //stores how many inhibitory connections each thread is creating
int *restrict SpkLiq_exc_connections_thread; //stores how many excitatory connections each thread is creating

int *restrict SpkLiq_number_of_inh_connections_thread; //stores how many inhibitory connections each thread will process
int *restrict SpkLiq_number_of_exc_connections_thread; //stores how many excitatory connections each thread will process

// SpikingLiquid_process_internal_spikes_threads
int *restrict SpkLiq_number_of_inh_neurons_thread; //stores the number of inhibitory neurons each thread is going to receive
int *restrict SpkLiq_number_of_exc_neurons_thread; //stores the number of excitatory neurons each thread is going to receive

int **restrict SpkLiq_receive_spike_i_thread; //stores the indices of which liquid's inhibitory neuron receives a spike (output from the threads)
float **restrict SpkLiq_receive_spike_i_w_thread; //stores the weights of the above received spikes
int *restrict SpkLiq_receive_spike_i_idx_thread; //stores how many items are inside each of the above arrays (each thread is going to generate an variable size)

int **restrict SpkLiq_receive_spike_e_thread; //stores the indices of which liquid's excitatory neuron receives a spike (output from the threads)
float **restrict SpkLiq_receive_spike_e_w_thread; //stores the weights of the above received spikes
int *restrict SpkLiq_receive_spike_e_idx_thread; //stores how many items are inside each of the above arrays (each thread is going to generate an variable size)


int SpkLiq_number_of_long_ints=0; //number of long ints necessary to map the whole network to bits only (used to load/save spikes)

int **restrict SpkLiq_neurons_inh_injections; //stores the indices of the inhibitory POS connections each neuron has.
int **restrict SpkLiq_neurons_exc_injections; //stores the indices of the excitatory POS connections each neuron has.

int *restrict SpkLiq_neurons_exc_injections_total; //stores the cummulative total number of connections according to the neurons index
int *restrict SpkLiq_neurons_inh_injections_total; //stores the cummulative total number of connections according to the neurons index

float **restrict SpkLiq_neurons_exc_injections_w; //stores the weights of the excitatory POS connections each neuron has.
float **restrict SpkLiq_neurons_inh_injections_w; //stores the weights of the inhibitory POS connections each neuron has.

int *restrict SpkLiq_number_of_inh_neurons_thread_total; // used to distribute the workload among threads
int *restrict SpkLiq_number_of_exc_neurons_thread_total; // used to distribute the workload among threads

int SpkLiq_initialized = 0; // mainly indicates to Python if the liquid's main variables are available
int SpkLiq_connected = 0; // mainly indicates to Python if the liquid's connections are available

// char *python_msgs; //used to send messages when using the simulator through Python

//
// Functions prototypes
void SpikingLiquid_generate_connections();
void *SpikingLiquid_generate_connections_less_memory_thread(void *args);
void SpikingLiquid_generate_connections_memory();
void SpikingLiquid_process_connections();
void *SpikingLiquid_Reset_thread(void *args);
void SpikingLiquid_Reset();
void SpikingLiquid_Soft_Reset(unsigned int *my_seeds);
void SpikingLiquid_init();
void SpkLiq_process_exc_spikes(const int *restrict spikes, const float *restrict weights, int const number_of_spikes);
void SpkLiq_process_inh_spikes(const int *restrict spikes, const float *restrict weights, int const number_of_spikes);
void *SpikingLiquid_process_internal_spikes_threads(void *args);
void *SpikingLiquid_process_internal_spikes_threads_new(void *args);
void *SpikingLiquid_update_internal_thread(void *args);
void SpikingLiquid_update(const int *restrict spikes_exc, const int *restrict spikes_inh, const float *restrict weights_exc, const float *restrict weights_inh, int const size_exc, int const size_inh);
void euler_distance_opt(float *restrict output, const int *restrict indices, int const loop_start, int const loop_end, int const number_of_neurons, float const lbd_value_opt, const Array3D* _3Didx);
int parse_sim_config(char *file_base_name);
int save_load_connections_file(char *file_base_name);


//
// Here come the functions!
//

void euler_distance_opt(float *restrict const results, const int *restrict const indices_ptr, int const loop_start, int const loop_end, int const number_of_neurons, float const lbd_value_opt, const Array3D*restrict const a_3Didx_ptr)
{
  /*
  This function calculates the distances altogether trying to give a hint to the compiler about what it could optmize.
  If memory becomes a problem, then this calculation could be done directly at the point where it is accessed.
    output <= distance_results
    indices <= SpkLiq_liquid_indices_shuffled
    number_of_neurons <= SpkLiq_number_of_neurons
    a_3Didx <= SpkLiq_3Didx
    lbd_value_opt <= SpkLiq_lbd_value_opt
  */

  // calculates for all the values acessible by the current thread
  for(int i=loop_start; i<loop_end; i++)
  {
    for(int j=0; j<number_of_neurons; j++)
    {
    results[(i-loop_start)*number_of_neurons+j]=exp(-(pow((a_3Didx_ptr[indices_ptr[i]].x-a_3Didx_ptr[indices_ptr[j]].x),2)+pow((a_3Didx_ptr[indices_ptr[i]].y-a_3Didx_ptr[indices_ptr[j]].y),2)+pow((a_3Didx_ptr[indices_ptr[i]].z-a_3Didx_ptr[indices_ptr[j]].z),2))*lbd_value_opt);
    }
  }
}


void SpikingLiquid_generate_connections()
{
  /*

  Starts the connection generation process...

  Here is also decided which neuron is inhibitory / excitatory.

  This function can only be called ONCE, because it frees some memory allocated by the SpikingLiquid_init()!

  */

    // Fills the structure with the correspondent values of the neuron indices [0]={0,0,0}; [1]={1,0,0}...
    // This is how the 3D structure of the liquid looks like.
    // The index of SpkLiq_3Didx is the index of the neuron.
    int count_3d=0;
    Array3D *restrict const SpkLiq_3Didx_internal = SpkLiq_3Didx; //trying to help gcc to optimize the code
    for(int zi=0; zi<SpkLiq_net_shape[2]; zi++)
    {
        for(int yi=0; yi<SpkLiq_net_shape[1]; yi++)
        {
            for(int xi=0; xi<SpkLiq_net_shape[0]; xi++)
            {
                SpkLiq_3Didx_internal[count_3d].x=xi;
                SpkLiq_3Didx_internal[count_3d].y=yi;
                SpkLiq_3Didx_internal[count_3d].z=zi;
                count_3d++;
            }
        }
    }


    //
    // Starts the shuffle of the liquid's neuron indices
    // This shuffle is necessary to select randomly the X% of neurons to be inhibitory.
    //
    // I NEED TO THINK MORE ABOUT THE BEST WAY TO SHUFFLE IN A MULTITHREAD WAY BECAUSE
    // RIGHT NOW I'M USING A SERIAL ALGORITHM THAT IS A BIT HARD TO MAKE PARALLEL...
    //
    const int SpkLiq_number_of_neurons_internal = SpkLiq_number_of_neurons; //trying to help gcc to optimize the code
    int *restrict const SpkLiq_liquid_indices_shuffled_internal = SpkLiq_liquid_indices_shuffled; //trying to help gcc to optimize the code

    for(int i=0; i<SpkLiq_number_of_neurons_internal; i++){
        SpkLiq_liquid_indices_shuffled_internal[i] = i; //Fills with integers from 0 to (SpkLiq_number_of_neurons-1) to be shuffled
    }

    //# RANDOM-2
    //# Selection of the inhibitory and excitatory neurons
    // Knuth-Fisher-Yates shuffle algorithm - from http://blog.codinghorror.com/the-danger-of-naivete/
    // Creates a shuffled array with all the liquid's indices
    rk_state *restrict const state_shuffle = &SpkLiq_threads_states[2][0]; //trying to help gcc to optimize the code
    for(int i=SpkLiq_number_of_neurons_internal - 1; i > 0; i--){
        // Generates a random integer from 0 to i+1
        int n = (int) (rk_double(state_shuffle)*(i+1)); // Passing the random state of thread 0, because this part is serial...

        int temp;

        // Swaps the current position 'i' with the random position 'n'
        temp = SpkLiq_liquid_indices_shuffled_internal[i],
        SpkLiq_liquid_indices_shuffled_internal[i] = SpkLiq_liquid_indices_shuffled_internal[n];
        SpkLiq_liquid_indices_shuffled_internal[n] = temp;
    }


    for(int i=0; i<SpkLiq_number_of_inh_neurons; i++){
        //Populates the SpkLiq_inhibitory_indices with the first N=SpkLiq_number_of_inh_neurons neurons from the shuffled liquid
        SpkLiq_inhibitory_indices[i] = SpkLiq_liquid_indices_shuffled[i];

        // //Populates the refractory_vetor with inhibitory values
        // SpkLiq_refrac_values[SpkLiq_inhibitory_indices[i]] = SpkLiq_refractory[1]; // Sets the refractory periods according to the type of neuron
        // Now these values are processed inside SpikingLiquid_process_connections() to make it possible to process user made connections
    }


    for(int i=SpkLiq_number_of_inh_neurons; i<SpkLiq_number_of_neurons; i++){
        //Populates the SpkLiq_excitatory_indices with the neurons that are not inhibitory
        SpkLiq_excitatory_indices[(i-SpkLiq_number_of_inh_neurons)] = SpkLiq_liquid_indices_shuffled[i];

        // //Populates the refractory_vetor with excitatory values
        // SpkLiq_refrac_values[SpkLiq_excitatory_indices[(i-SpkLiq_number_of_inh_neurons)]] = SpkLiq_refractory[0]; // Sets the refractory periods according to the type of neuron
        // Now these values are processed inside SpikingLiquid_process_connections() to make it possible to process user made connections
    }


    printf("Generating liquid's connections...\n");

    // This is going to be used to control the memory positions each thread can access.
    int SpkLiq_number_of_neurons_in_slice_init = SpkLiq_number_of_neurons/SpkLiq_threads_N;
    if (SpkLiq_number_of_neurons_in_slice_init<2)
    {
      fprintf(stderr,"SpikingLiquid_generate_connections: SpkLiq_number_of_neurons_in_slice_init - too many threads (too few neurons)!\n");
      exit(EXIT_FAILURE);
    }

    MAX_CONNECTIONS = SpkLiq_number_of_neurons_in_slice_init; //I'm supposing a good value for the minimum number of connections is the
                                                              //same as the number of neurons in the slice

//
// MULTITHREADS CODE
//
    // Creates the threads to calculate the probabilities based on the distance between neurons
    // and generate the connections between the inhibitory and excitatoty neurons
    for (int i = 0; i < SpkLiq_threads_N; ++i) {
        if (pthread_create(&SpkLiq_threads_ptrs[i], NULL, SpikingLiquid_generate_connections_less_memory_thread, &SpkLiq_thread_id[i])) {
          fprintf(stderr, "error: SpikingLiquid_generate_connections_less_memory_thread\n");
          exit(EXIT_FAILURE);
        }
    }
    // Block (join) all threads - the program is stuck here until all threads return
    for (int i = 0; i < SpkLiq_threads_N; ++i) {
        pthread_join(SpkLiq_threads_ptrs[i], NULL);
    }


    printf("Copying connections...\n");

    // Groups the connections generated by each thread in a single array (for inhibitory and excitatory separately)
    SpikingLiquid_generate_connections_memory(); // to copy the memory content is WAY FASTER than the generation of connections!

}

void *SpikingLiquid_generate_connections_less_memory_thread(void *args)
{

    /*
    Generates the liquid's probabilistic small world network structure.
    http://en.wikipedia.org/wiki/Small-world_network
    "Small-world brain networks" - http://www.ncbi.nlm.nih.gov/pubmed/17079517

    Stores the liquid's structure in the following arrays (each index "thread_id" represents one thread):
    (at the end the data must be brought together becasue which thread generates a set of separated arrays)
    SpkLiq_pre_i_thread[thread_id]: indices of the inhibitory PREsynaptic connections
    SpkLiq_pos_i_thread[thread_id]: indices of the inhibitory POSsynaptic connections
    SpkLiq_w_i_thread[thread_id]: weights of the inhibitory=>? connections

    SpkLiq_pre_e_thread[thread_id]: indices of the excitatory PREsynaptic connections
    SpkLiq_pos_e_thread[thread_id]: indices of the excitatory POSsynaptic connections
    SpkLiq_w_e_thread[thread_id]: weights of the excitatory=>? connections

    SpkLiq_inh_connections_thread[thread_id]: total number of INHIBITORY connections generated by the thread
    SpkLiq_exc_connections_thread[thread_id]: total number of EXCITATORY connections generated by the thread
    */

    const int SpkLiq_tid = *((int *)args); //Receives the number of the thread from the function input arguments

    int PRE; //Flags to indicate the type of connection: inhibitory=>0 and excitatory=>1
    int POS; //during the double for loop where the connections are generated.

    int n_malloc_i = 1; //Counter used with realloc to dynamically allocate more memory: inhibitory connections
    int n_malloc_e = 1; //Counter used with realloc to dynamically allocate more memory: excitatory connections

    int loop_start, loop_end;

    int *restrict const SpkLiq_inh_connections_thread_internal = &SpkLiq_inh_connections_thread[SpkLiq_tid]; //trying to help gcc to optimize the code
    int *restrict const SpkLiq_exc_connections_thread_internal = &SpkLiq_exc_connections_thread[SpkLiq_tid]; //trying to help gcc to optimize the code

    int *restrict SpkLiq_pre_i_thread_internal;  //SpkLiq_pre_i_thread[SpkLiq_tid]
    int *restrict SpkLiq_pos_i_thread_internal;  //SpkLiq_pos_i_thread[SpkLiq_tid]
    float *restrict SpkLiq_w_i_thread_internal; //SpkLiq_w_i_thread[SpkLiq_tid]

    int *restrict SpkLiq_pre_e_thread_internal;  //SpkLiq_pre_e_thread[SpkLiq_tid]
    int *restrict SpkLiq_pos_e_thread_internal;  //SpkLiq_pos_e_thread[SpkLiq_tid]
    float *restrict SpkLiq_w_e_thread_internal; //SpkLiq_w_e_thread[SpkLiq_tid]

    const int *restrict const SpkLiq_liquid_indices_shuffled_internal = SpkLiq_liquid_indices_shuffled; //trying to help gcc to optimize the code

    const int MAX_CONNECTIONS_internal = MAX_CONNECTIONS; //trying to help gcc to optimize the code

    const int SpkLiq_number_of_inh_neurons_internal = SpkLiq_number_of_inh_neurons; //trying to help gcc to optimize the code

    const int SpkLiq_number_of_neurons_internal = SpkLiq_number_of_neurons; //trying to help gcc to optimize the code

    int *restrict const SpkLiq_neurons_connected_internal = SpkLiq_neurons_connected; //trying to help gcc to optimize the code

    const Array3D *restrict const a_3Didx_ptr = SpkLiq_3Didx ;
    const int *restrict const indices_ptr = SpkLiq_liquid_indices_shuffled_internal;
    const float lbd_value_opt = SpkLiq_lbd_value_opt;

    // Initializes the number of connections (inh / exc)'s counters
    *SpkLiq_inh_connections_thread_internal = 0;
    *SpkLiq_exc_connections_thread_internal = 0;

    //
    //Now each thread is going to have its own arrays and acesses using the thread index "SpkLiq_tid"
    //
    //Allocate memory to save the inhibitory=>? connections
    SpkLiq_pre_i_thread_internal = malloc(MAX_CONNECTIONS_internal*sizeof(int));
    SpkLiq_pos_i_thread_internal = malloc(MAX_CONNECTIONS_internal*sizeof(int));
    SpkLiq_w_i_thread_internal = malloc(MAX_CONNECTIONS_internal*sizeof(float));

    //Allocate memory to save the excitatory=>? connections
    SpkLiq_pre_e_thread_internal = malloc(MAX_CONNECTIONS_internal*sizeof(int));
    SpkLiq_pos_e_thread_internal = malloc(MAX_CONNECTIONS_internal*sizeof(int));
    SpkLiq_w_e_thread_internal = malloc(MAX_CONNECTIONS_internal*sizeof(float));

    if (SpkLiq_pre_i_thread_internal==NULL || SpkLiq_pos_i_thread_internal==NULL || SpkLiq_w_i_thread_internal==NULL || SpkLiq_pre_e_thread_internal==NULL || SpkLiq_pos_e_thread_internal==NULL || SpkLiq_w_e_thread_internal==NULL)
    {
        fprintf(stderr,"connections memory malloc ERROR!");
        exit(EXIT_FAILURE); //I need to check which error I should signal here...
    }

    // Each thread receives a slice of the total number of neurons.
    // but if the result of the division of total number of neurons by number of threads
    // has a remainder, then the last thread will have a bigger set of values to include the remainder.

    // Sets up the loop variables according to the thread number
    if (SpkLiq_tid!=(SpkLiq_threads_N-1))
    {
      loop_start = SpkLiq_tid*SpkLiq_number_of_neurons_slice[SpkLiq_tid];
      loop_end = (SpkLiq_number_of_neurons_slice[SpkLiq_tid])*(SpkLiq_tid+1);
    }else
    { // It means the last thread that could have more work to do because of the remainder
      loop_start = SpkLiq_tid*SpkLiq_number_of_neurons_slice[SpkLiq_tid-1]; //Uses the anterior value (the rounded one) and multiplies
      loop_end = (SpkLiq_number_of_neurons_slice[SpkLiq_tid-1])*SpkLiq_tid+(SpkLiq_number_of_neurons_slice[SpkLiq_tid]);
    }

    /*
        PRE  POS
    a) II : 0    0
    b) IE : 0    1
    c) EE : 1    1
    d) EI : 1    0

    if(index<SpkLiq_number_of_inh_neurons_internal) => I
    if(index>=SpkLiq_number_of_inh_neurons_internal) => E

    */

    int const loop_start_internal = loop_start; //trying to help gcc to optimize the code
    int const loop_end_internal = loop_end; //trying to help gcc to optimize the code
    // Goes through all the neurons indices and verify if they are inhibitory (excitatory test is implicit)
    // Instead of using the ordered list, I'm going to use the shuffled one because this way I don't need to check
    // which neuron is excitatory or inhibitory in a difficult (cpu costly) way.
    for (int i=loop_start_internal; i<loop_end_internal; i++)
    { //These are the PRE indices
        PRE = 1; //Because it's only tested against inhibitory indices

        // Verifies if the neuron index 'i' belongs to SpkLiq_inhibitory_indices
        if(i<SpkLiq_number_of_inh_neurons_internal)
        {
            PRE = 0; //Indicates the PRE-synaptic neuron is inhibitory
        }


        for (int j=0;j<SpkLiq_number_of_neurons_internal; j++)
        { //These are the POS indices
            POS = 1;//Because it's only tested against inhibitory indices

            // Verifies if the neurons indices 'i' and 'j' belongs to SpkLiq_inhibitory_indices
            if(j<SpkLiq_number_of_inh_neurons_internal)
            {
                POS = 0; //Indicates the POS-synaptic neuron is inhibitory
            }


            //Here the variables about PRE/POS are set

            // This test is the most expensive calculation in this loop
            // According to my tests, the nested for loops are about 170 faster than only this calculation.
            // Also, to allocate memory to store the results of ALL possible combinations is super fast.
            // So I should do this calculation in parallel, storing the results in memory.
            // After that I couldl do one thread to generate the vectors SpkLiq_pre_i_thread, ...
            // or even think about a more optimized way because, after all, the update function is VERY important too.

            // Verifies if a connection (synapse) will be created or not depending on the calculated probability.
            // # RANDOM-3
            // # Internal connections of the liquid
            //

            //This lines guarantee that rk_double will be called ONLY if the probability based on distance is bigger than SpkLiq_min_perc
            float const prob_dist = (SpkLiq_liquid_parameters[PRE][POS][0]*exp(-(pow((a_3Didx_ptr[indices_ptr[i]].x-a_3Didx_ptr[indices_ptr[j]].x),2)+pow((a_3Didx_ptr[indices_ptr[i]].y-a_3Didx_ptr[indices_ptr[j]].y),2)+pow((a_3Didx_ptr[indices_ptr[i]].z-a_3Didx_ptr[indices_ptr[j]].z),2))*lbd_value_opt));
            int creation_test = 0;
            if(prob_dist>=SpkLiq_min_perc)
                creation_test = (float)rk_double(&SpkLiq_threads_states[3][SpkLiq_tid])<=prob_dist;

            if (creation_test)
            {
                //It means we have a synapse here!

                if(i!=j)
                {
                  //Signalizes that this neuron has at least ONE connection to other neuron
                  SpkLiq_neurons_connected_internal[SpkLiq_liquid_indices_shuffled_internal[i]]=1;
                  SpkLiq_neurons_connected_internal[SpkLiq_liquid_indices_shuffled_internal[j]]=1; //This is not thread safe, but
                                                                                                   //is not a problem because they are going to
                                                                                                   //write the same value!!!
                }

                if(PRE==0)
                {//Inhibitory=>? connection
                    (*SpkLiq_inh_connections_thread_internal)++;

                    //Verifies if the array (used to store the Inhibitory=>? connections) needs more memory
                    if ((*SpkLiq_inh_connections_thread_internal)>(MAX_CONNECTIONS_internal*n_malloc_i)){
                        n_malloc_i++;
                        SpkLiq_pre_i_thread_internal = realloc(SpkLiq_pre_i_thread_internal,n_malloc_i*MAX_CONNECTIONS_internal*sizeof(int));
                        SpkLiq_pos_i_thread_internal = realloc(SpkLiq_pos_i_thread_internal,n_malloc_i*MAX_CONNECTIONS_internal*sizeof(int));
                        SpkLiq_w_i_thread_internal = realloc(SpkLiq_w_i_thread_internal,n_malloc_i*MAX_CONNECTIONS_internal*sizeof(float));
                        if (SpkLiq_pre_i_thread_internal==NULL || SpkLiq_pos_i_thread_internal==NULL || SpkLiq_w_i_thread_internal==NULL){
                            fprintf(stderr,"SpikingLiquid_generate_connections INH - realloc ERROR!");
                            exit(EXIT_FAILURE);
                        }
                    }

                    SpkLiq_pre_i_thread_internal[(*SpkLiq_inh_connections_thread_internal)-1]=SpkLiq_liquid_indices_shuffled_internal[i];
                    SpkLiq_pos_i_thread_internal[(*SpkLiq_inh_connections_thread_internal)-1]=SpkLiq_liquid_indices_shuffled_internal[j];
                 // SpkLiq_w_i_thread_internal[(*SpkLiq_inh_connections_thread_internal)-1]=1E-9*SpkLiq_liquid_parameters[PRE][POS][4]*fabs((float)rk_gauss(&SpkLiq_threads_states[3][SpkLiq_tid])*0.5+1); //AMaass
                    SpkLiq_w_i_thread_internal[(*SpkLiq_inh_connections_thread_internal)-1]=1E-9*SpkLiq_liquid_parameters[PRE][POS][4]*fabs((float)rk_normal(&SpkLiq_threads_states[3][SpkLiq_tid],0,1)*0.5+1); //AMaass

                }else
                {//Excitatory=>? connections
                    (*SpkLiq_exc_connections_thread_internal)++;

                    //Verifies if the array (used to store the Excitatory=>? connections) needs more memory
                    if ((*SpkLiq_exc_connections_thread_internal)>(MAX_CONNECTIONS_internal*n_malloc_e)){
                        n_malloc_e++;
                        SpkLiq_pre_e_thread_internal = realloc(SpkLiq_pre_e_thread_internal,n_malloc_e*MAX_CONNECTIONS_internal*sizeof(int));
                        SpkLiq_pos_e_thread_internal = realloc(SpkLiq_pos_e_thread_internal,n_malloc_e*MAX_CONNECTIONS_internal*sizeof(int));
                        SpkLiq_w_e_thread_internal = realloc(SpkLiq_w_e_thread_internal,n_malloc_e*MAX_CONNECTIONS_internal*sizeof(float));
                        if (SpkLiq_pre_e_thread_internal==NULL || SpkLiq_pos_e_thread_internal==NULL || SpkLiq_w_e_thread_internal==NULL){
                            fprintf(stderr,"SpikingLiquid_generate_connections EXC - realloc ERROR!");
                            exit(EXIT_FAILURE);
                        }
                    }

                    SpkLiq_pre_e_thread_internal[(*SpkLiq_exc_connections_thread_internal)-1]=SpkLiq_liquid_indices_shuffled_internal[i];
                    SpkLiq_pos_e_thread_internal[(*SpkLiq_exc_connections_thread_internal)-1]=SpkLiq_liquid_indices_shuffled_internal[j];
                 // SpkLiq_w_e_thread_internal[(*SpkLiq_exc_connections_thread_internal)-1]=1E-9*SpkLiq_liquid_parameters[PRE][POS][4]*fabs((float)rk_gauss(&SpkLiq_threads_states[3][SpkLiq_tid])*0.5+1); //AMaass
                    SpkLiq_w_e_thread_internal[(*SpkLiq_exc_connections_thread_internal)-1]=1E-9*SpkLiq_liquid_parameters[PRE][POS][4]*fabs((float)rk_normal(&SpkLiq_threads_states[3][SpkLiq_tid],0,1)*0.5+1); //AMaass
                }
            }

        }
    }

    // Passes the allocated memory positions
    SpkLiq_pre_i_thread[SpkLiq_tid] = SpkLiq_pre_i_thread_internal;
    SpkLiq_pos_i_thread[SpkLiq_tid] = SpkLiq_pos_i_thread_internal;
    SpkLiq_w_i_thread[SpkLiq_tid] = SpkLiq_w_i_thread_internal;

    SpkLiq_pre_e_thread[SpkLiq_tid]=SpkLiq_pre_e_thread_internal;
    SpkLiq_pos_e_thread[SpkLiq_tid]=SpkLiq_pos_e_thread_internal;
    SpkLiq_w_e_thread[SpkLiq_tid]=SpkLiq_w_e_thread_internal;

    return NULL;
}


void SpikingLiquid_generate_connections_memory()
{

    /*
    Copies the data generated by the SpikingLiquid_generate_connections_less_memory_thread into unique arrays.
    Moving to a continuous segment of memory is going to make the simulation's update faster and avoid extra code
    necessary to deal with a diffuse memory model.

    In a system with less memory, maybe it is going to be necessary to break the memory copy into small chunks to make
    it possible to free memory just after it had been copied (maybe using some realloc calls too).

    Because this memory copy is very fast, I decided to keep it in a serial code instead of using threads.
    */

    // Sums up the totals generated by the SpikingLiquid_generate_connections_less_memory_thread
    // This is necessary because we need to allocate the memory for the variables.
    SpkLiq_inh_connections = 0;
    SpkLiq_exc_connections = 0;
    for(int i=0; i<SpkLiq_threads_N; i++)
    {
            SpkLiq_inh_connections+=SpkLiq_inh_connections_thread[i];
            SpkLiq_exc_connections+=SpkLiq_exc_connections_thread[i];
    }

    // Debug
    // printf("Debug - SpkLiq_inh_connections:%d\n", SpkLiq_inh_connections);
    // printf("Debug - SpkLiq_exc_connections:%d\n", SpkLiq_exc_connections);

    // Allocates the memory according to the totals generated above
    SpkLiq_pre_i = malloc(sizeof(int)*SpkLiq_inh_connections);
    SpkLiq_pos_i = malloc(sizeof(int)*SpkLiq_inh_connections);
    SpkLiq_w_i = malloc(sizeof(float)*SpkLiq_inh_connections);

    SpkLiq_pre_e = malloc(sizeof(int)*SpkLiq_exc_connections);
    SpkLiq_pos_e = malloc(sizeof(int)*SpkLiq_exc_connections);
    SpkLiq_w_e = malloc(sizeof(float)*SpkLiq_exc_connections);

    if (SpkLiq_pre_i==NULL || SpkLiq_pos_i==NULL || SpkLiq_w_i==NULL || SpkLiq_pre_e==NULL || SpkLiq_pos_e==NULL || SpkLiq_w_e==NULL)
    {
        fprintf(stderr,"SpikingLiquid_generate_connections_memory -  malloc ERROR!");
        exit(EXIT_FAILURE); //I need to check which error I should signal here...
    }


    // Effectively moves the data to the continuous positions allocated above
    int index_conter_i=0;
    int index_conter_e=0;
    for(int i=0; i<SpkLiq_threads_N; i++)
    { // iterates through the threads
        for(int j=0; j<SpkLiq_inh_connections_thread[i]; j++)
        { // iterates through the arrays generated by each thread
            if (index_conter_i<SpkLiq_inh_connections)
            {
              SpkLiq_pre_i[index_conter_i] = SpkLiq_pre_i_thread[i][j];
              SpkLiq_pos_i[index_conter_i] = SpkLiq_pos_i_thread[i][j];
              SpkLiq_w_i[index_conter_i] = SpkLiq_w_i_thread[i][j];
              index_conter_i++;
            }else{
              fprintf(stderr,"index_conter_i overflow!");
              exit(EXIT_FAILURE); //I need to check which error I should signal here...
            }

        }
        // Frees the memory used by these variables!
        free(SpkLiq_pre_i_thread[i]);
        free(SpkLiq_pos_i_thread[i]);
        free(SpkLiq_w_i_thread[i]);

        for(int j=0; j<SpkLiq_exc_connections_thread[i]; j++){
          if (index_conter_e<SpkLiq_exc_connections)
          {
            SpkLiq_pre_e[index_conter_e] = SpkLiq_pre_e_thread[i][j];
            SpkLiq_pos_e[index_conter_e] = SpkLiq_pos_e_thread[i][j];
            SpkLiq_w_e[index_conter_e] = SpkLiq_w_e_thread[i][j];
            index_conter_e++;
          }else{
            fprintf(stderr,"index_conter_e overflow!");
            exit(EXIT_FAILURE); //I need to check which error I should signal here...
          }
        }
        // Frees the memory used by these variables!
        free(SpkLiq_pre_e_thread[i]);
        free(SpkLiq_pos_e_thread[i]);
        free(SpkLiq_w_e_thread[i]);
    }
        free(SpkLiq_pre_i_thread);
        free(SpkLiq_pos_i_thread);
        free(SpkLiq_w_i_thread);

        free(SpkLiq_pre_e_thread);
        free(SpkLiq_pos_e_thread);
        free(SpkLiq_w_e_thread);

        free(SpkLiq_inh_connections_thread);
        free(SpkLiq_exc_connections_thread);


}


void SpikingLiquid_process_connections()
{
  /*
    Processes the connections in order to set up everything to make the update process possible!

    If the user wants to create connections, it's necessary to populate:
    - SpkLiq_inhibitory_indices: array with the neuron's indices belonging to inhibitory group
    - SpkLiq_excitatory_indices: array with the neuron's indices belonging to excitatory group

    - SpkLiq_inh_connections: total number of inhibitory=>? connections
    - SpkLiq_exc_connections: total number of excitatory=>? connections

    - SpkLiq_pre_i: stores the indices of the inhibitory=>? PRE-synaptic connections in the liquid
    - SpkLiq_pos_i: stores the indices of the inhibitory=>? POS-synaptic connections in the liquid
    - SpkLiq_w_i: stores the weights of the inhibitory connections above

    - SpkLiq_pre_e: stores the indices of the excitatory=>? PRE-synaptic connections in the liquid
    - SpkLiq_pos_e: stores the indices of the excitatory=>? POS-synaptic connections in the liquid
    - SpkLiq_w_e: stores the weights of the excitatory connections above

  */

  //////////////////////////////////////////////////////
  //////////////NEW CONNECTIONS ORGANIZER
  //////////////////////////////////////////////////////
  /*
  SpkLiq_neurons_exc_injections[i] => has an array with all the neurons receiving EXCITATORY spikes from 'i'
  SpkLiq_neurons_exc_injections[i][0] => has the number of itens (connections) the array above has
  SpkLiq_neurons_exc_injections_w[i] => has an array with all the weights from the connections above
  SpkLiq_neurons_exc_injections_total => has the cumulative number of connections

  SpkLiq_neurons_inh_injections[i] => has an array with all the neurons receiving INHIBITORY spikes from 'i'
  SpkLiq_neurons_inh_injections[i][0] => has the number of itens (connections) the array above has
  SpkLiq_neurons_inh_injections_w[i] => has an array with all the weights from the connections above
  SpkLiq_neurons_inh_injections_total => has the cumulative number of connections

  SpkLiq_neurons_exc/inh_injections[i] => if the neurons has NO connections, then instead of an array there is a NULL pointer!

  It's possible to divide the workload among the threads using the information available about how many POS connections each neuron has!!!

  Now I need to check how to divide the update workload among the threads and maybe how to make the function that process the spikes multithreaded!!!
  */
    for(int i=0; i<SpkLiq_number_of_neurons; i++) // This goes through ALL the neurons, therefore the arrays below will be sparse!
    {
      SpkLiq_neurons_exc_injections[i]=NULL; //First fills everything with NULL pointer, hence if there is a NULL pointer means that position is not excitatory
      SpkLiq_neurons_inh_injections[i]=NULL; //First fills everything with NULL pointer, hence if there is a NULL pointer means that position is not inhibitory
      SpkLiq_neurons_exc_injections_w[i]=NULL;
      SpkLiq_neurons_inh_injections_w[i]=NULL;
    }

    // Creates the memory positions to the inhibitory injections
    // This is a waste of memory, but avoids to start doing mallocs inside the loop that really organizes the memory
    for(int i=0; i<SpkLiq_number_of_inh_neurons; i++)
    {
      const int idx = SpkLiq_inhibitory_indices[i];
      SpkLiq_neurons_inh_injections[idx]=malloc(sizeof(int)*(MAX_INDIVIDUAL_CONNECTIONS+1)); //the '+1' is for the '[0]'
      SpkLiq_neurons_inh_injections_w[idx]=malloc(sizeof(float)*(MAX_INDIVIDUAL_CONNECTIONS+1));
      if ((SpkLiq_neurons_inh_injections[idx]==NULL) || (SpkLiq_neurons_inh_injections_w[idx]==NULL))
      {
        fprintf(stderr,"SpkLiq_neurons_inh_injections/_w[%d] - malloc error\n",idx);
        exit(EXIT_FAILURE);
      }
      SpkLiq_neurons_inh_injections[idx][0]=0; //The first index stores the total number of connections
                                               //Because of that the realloc must always allocate an extra position for this value!!!
    }


    // Creates the starting memory positions to the excitatory injections
    // This is a waste of memory, but avoids to start doing mallocs inside the loop that really organizes the memory
    for(int i=0; i<(SpkLiq_number_of_neurons-SpkLiq_number_of_inh_neurons); i++)
    {
      const int idx = SpkLiq_excitatory_indices[i];
      SpkLiq_neurons_exc_injections[idx]=malloc(sizeof(int)*(MAX_INDIVIDUAL_CONNECTIONS+1)); //the '+1' is for the '[0]'
      SpkLiq_neurons_exc_injections_w[idx]=malloc(sizeof(float)*(MAX_INDIVIDUAL_CONNECTIONS+1));
      if ((SpkLiq_neurons_exc_injections[idx]==NULL) || (SpkLiq_neurons_exc_injections_w[idx]==NULL))
      {
        fprintf(stderr,"SpkLiq_neurons_exc_injections/_w[%d] - malloc error\n",idx);
        exit(EXIT_FAILURE);
      }
      SpkLiq_neurons_exc_injections[idx][0]=0; //The first index stores the total number of connections
                                               //Because of that the realloc must always allocate an extra position for this value!!!
    }



    // Organizes the inhibitory memory positions
    int *memory_space = calloc(SpkLiq_number_of_neurons,sizeof(int)); //used to control the realloc
                                                                      //initialized with ZERO
    for(int i=0; i<SpkLiq_inh_connections; i++)
    {
        const int idx_pre = SpkLiq_pre_i[i];
        const int idx_pos = SpkLiq_pos_i[i];
        const float weight = SpkLiq_w_i[i];
        int *restrict const idx_counter = &SpkLiq_neurons_inh_injections[idx_pre][0];

        // NEW INTERNAL CONNECTIONS
        if(weight != 0.0)
        {
          SpkLiq_neurons_inh_injections[idx_pre][(*idx_counter)+1]=idx_pos; //stores the pos connections
                                                                            //the '+1' is to jump over the first position '[0]'
          SpkLiq_neurons_inh_injections_w[idx_pre][(*idx_counter)+1]=weight; //stores the pos connections weight
          (*idx_counter)++; //adds to the connection counter, remember this counter starts with ZERO
          if((*idx_counter) == ((memory_space[idx_pre]+1)*MAX_INDIVIDUAL_CONNECTIONS)) //initially the memory is MAX_INDIVIDUAL_CONNECTIONS
          {
            memory_space[idx_pre]++;

            SpkLiq_neurons_inh_injections[idx_pre] = realloc(SpkLiq_neurons_inh_injections[idx_pre],((memory_space[idx_pre]+1)*MAX_INDIVIDUAL_CONNECTIONS+1)*sizeof(int)); //the '+1' is to add space for the '[0]'
            if (SpkLiq_neurons_inh_injections[idx_pre]==NULL)
            {
              fprintf(stderr,"SpkLiq_neurons_inh_injections[%d] - memory_space[%d]+1:%d - realloc error\n",idx_pre,idx_pre,memory_space[idx_pre]+1);
              exit(EXIT_FAILURE);
            }

            SpkLiq_neurons_inh_injections_w[idx_pre] = realloc(SpkLiq_neurons_inh_injections_w[idx_pre],((memory_space[idx_pre]+1)*MAX_INDIVIDUAL_CONNECTIONS+1)*sizeof(float)); //the '+1' is to add space for the '[0]'
            if (SpkLiq_neurons_inh_injections_w[idx_pre]==NULL)
            {
              fprintf(stderr,"SpkLiq_neurons_inh_injections_w[%d] - memory_space[%d]+1:%d - realloc error\n",idx_pre,idx_pre,memory_space[idx_pre]+1);
              exit(EXIT_FAILURE);
            }

          }
        }
    }
    free(memory_space);


    // Fills the SpkLiq_neurons_inh_injections_total with the cumulative total number of connections
    int total_inh_connections = 0;
    int idx_i = 0;
    for(int i=0; i<SpkLiq_number_of_neurons; i++) //goes through ALL the neurons
    {
      if(SpkLiq_neurons_inh_injections[i]!=NULL) //verifies if this particular inhibitory neuron sends spikes to another one
      {
        total_inh_connections+=SpkLiq_neurons_inh_injections[i][0];
        SpkLiq_inhibitory_indices[idx_i++]=i; //puts the indices in order
      }

      SpkLiq_neurons_inh_injections_total[i]=total_inh_connections;

      // Debug
      // printf("Debug - SpkLiq_neurons_inh_injections_total[%d]:%d\n",i,total_inh_connections);
    }


    // Organizes the excitatory memory positions
    memory_space = calloc(SpkLiq_number_of_neurons,sizeof(int)); //used to control the realloc
                                                                 //initialized with ZERO
    for(int i=0; i<SpkLiq_exc_connections; i++)
    {
        const int idx_pre = SpkLiq_pre_e[i];
        const int idx_pos = SpkLiq_pos_e[i];
        const float weight = SpkLiq_w_e[i];
        int *restrict const idx_counter = &SpkLiq_neurons_exc_injections[idx_pre][0];

        // NEW INTERNAL CONNECTIONS
        if(weight != 0.0)
        {
          SpkLiq_neurons_exc_injections[idx_pre][*idx_counter+1]=idx_pos; //stores the pos connections
                                                                          //the '+1' is to jump over the first position '[0]'
          SpkLiq_neurons_exc_injections_w[idx_pre][*idx_counter+1]=weight; //stores the pos connections
          (*idx_counter)++; //adds to the connection counter, remember this counter starts with ZERO
          if((*idx_counter) == ((memory_space[idx_pre]+1)*MAX_INDIVIDUAL_CONNECTIONS))
          {
            memory_space[idx_pre]++;
            SpkLiq_neurons_exc_injections[idx_pre] = realloc(SpkLiq_neurons_exc_injections[idx_pre],((memory_space[idx_pre]+1)*MAX_INDIVIDUAL_CONNECTIONS+1)*sizeof(int)); //the '+1' is to add space for the '[0]'
            if (SpkLiq_neurons_exc_injections[idx_pre]==NULL)
            {
              fprintf(stderr,"SpkLiq_neurons_exc_injections[%d] - memory_space[%d]+1:%d - realloc error\n",idx_pre,idx_pre,memory_space[idx_pre]+1);
              exit(EXIT_FAILURE);
            }
            SpkLiq_neurons_exc_injections_w[idx_pre] = realloc(SpkLiq_neurons_exc_injections_w[idx_pre],((memory_space[idx_pre]+1)*MAX_INDIVIDUAL_CONNECTIONS+1)*sizeof(float)); //the '+1' is to add space for the '[0]'
            if (SpkLiq_neurons_exc_injections_w[idx_pre]==NULL)
            {
              fprintf(stderr,"SpkLiq_neurons_exc_injections_w[%d] - memory_space[%d]+1:%d - realloc error\n",idx_pre,idx_pre,memory_space[idx_pre]+1);
              exit(EXIT_FAILURE);
            }
          }
        }
    }
    free(memory_space);

    // Fills the SpkLiq_neurons_exc_injections_total with the cumulative total number of connections
    // With this information is possible to know exactly the maximum amount of memory each thread could need
    int total_exc_connections = 0;
    int idx_e = 0;
    for(int i=0; i<SpkLiq_number_of_neurons; i++)
    {
      if(SpkLiq_neurons_exc_injections[i]!=NULL)
      {
        total_exc_connections+=SpkLiq_neurons_exc_injections[i][0];
        SpkLiq_excitatory_indices[idx_e++]=i; //puts the indices in order
      }

      SpkLiq_neurons_exc_injections_total[i]=total_exc_connections;
    }

    // Now SpkLiq_inhibitory_indices and SpkLiq_excitatory_indices are ascending ordered!!!!
    //
//////////////////////////////////////////////////////
//////////////NEW CONNECTIONS ORGANIZER - END
//////////////////////////////////////////////////////


    //
    // Setting up the refractory arrays
    //
    for(int i=0; i<SpkLiq_number_of_inh_neurons; i++)
    {
        //Populates the refractory_vetor with inhibitory values
        SpkLiq_refrac_values[SpkLiq_inhibitory_indices[i]] = SpkLiq_refractory[1]; // Sets the refractory periods according to the type of neuron
        // Debug
        // printf("Debug - [INH]SpkLiq_refrac_values[%d]:%f\n",SpkLiq_inhibitory_indices[i],SpkLiq_refrac_values[SpkLiq_inhibitory_indices[i]]*1E6);
    }

    for(int i=SpkLiq_number_of_inh_neurons; i<SpkLiq_number_of_neurons; i++)
    {
        //Populates the refractory_vetor with excitatory values
        SpkLiq_refrac_values[SpkLiq_excitatory_indices[(i-SpkLiq_number_of_inh_neurons)]] = SpkLiq_refractory[0]; // Sets the refractory periods according to the type of neuron
        // Debug
        // printf("Debug - [EXC]SpkLiq_refrac_values[%d]:%f\n",SpkLiq_excitatory_indices[(i-SpkLiq_number_of_inh_neurons)],SpkLiq_refrac_values[SpkLiq_excitatory_indices[(i-SpkLiq_number_of_inh_neurons)]]*1E6);
    }




    //
    // Initializes the arrays that will be used during the update with the threads
    //

    // Spreads the inh/exc neurons among the threads
    // Allocates a memory position to each thread to store the number of neurons it will process
    SpkLiq_number_of_inh_neurons_thread = malloc(sizeof(int)*SpkLiq_threads_N);
    SpkLiq_number_of_exc_neurons_thread = malloc(sizeof(int)*SpkLiq_threads_N);

    if (SpkLiq_number_of_inh_neurons_thread==NULL || SpkLiq_number_of_exc_neurons_thread==NULL){
        fprintf(stderr, "SpkLiq_number_of_(inh/exc)_neurons_thread - malloc error!\n");
        exit(EXIT_FAILURE);
    }

    int SpkLiq_number_of_inh_neurons_slice = SpkLiq_number_of_inh_neurons/SpkLiq_threads_N;
    if (SpkLiq_number_of_inh_neurons_slice<2)
    {
      fprintf(stderr,"SpikingLiquid_process_connections: SpkLiq_number_of_neurons_in_slice - too many threads (too few inh neurons)!\n");
      exit(EXIT_FAILURE);
    }

    int SpkLiq_number_of_exc_neurons_slice = SpkLiq_number_of_exc_neurons/SpkLiq_threads_N;
    if (SpkLiq_number_of_exc_neurons_slice<2)
    {
      fprintf(stderr,"SpikingLiquid_process_connections: SpkLiq_number_of_neurons_in_slice - too many threads (too few exc neurons)!\n");
      exit(EXIT_FAILURE);
    }

    //
    // Spreads the liquid's inh/exc neurons among the threads
    //
    for(int i=0; i<SpkLiq_threads_N; i++){
        // And here how many neurons (inhibitory / excitatory) each thread is going to receive
        SpkLiq_number_of_inh_neurons_thread[i] = SpkLiq_number_of_inh_neurons_slice;
        SpkLiq_number_of_exc_neurons_thread[i] = SpkLiq_number_of_exc_neurons_slice;
    }

    if (SpkLiq_number_of_inh_neurons % SpkLiq_threads_N != 0)
    { // Verifies if there's a remainder
        // In the case of a remainder, makes the last thread heavier adding the remainder
        SpkLiq_number_of_inh_neurons_thread[SpkLiq_threads_N-1] = SpkLiq_number_of_inh_neurons_slice + (SpkLiq_number_of_inh_neurons % SpkLiq_threads_N);
    }

    if (SpkLiq_number_of_exc_neurons % SpkLiq_threads_N != 0)
    { // Verifies if there's a remainder
        // In the case of a remainder, makes the last thread heavier adding the remainder
        SpkLiq_number_of_exc_neurons_thread[SpkLiq_threads_N-1] = SpkLiq_number_of_exc_neurons_slice + (SpkLiq_number_of_exc_neurons % SpkLiq_threads_N);
    }

    //
    // After this point is important to know how many connections each thread is going to process, otherwise
    // it's not possible to allocate the right minimum amount of memory.
    //

    // Generates cumulative lists to make easier the spread of workload among threads!
    SpkLiq_number_of_inh_neurons_thread_total = calloc(SpkLiq_threads_N,sizeof(int));
    SpkLiq_number_of_exc_neurons_thread_total = calloc(SpkLiq_threads_N,sizeof(int));
    int SpkLiq_number_of_inh_neurons_thread_sum = 0;
    int SpkLiq_number_of_exc_neurons_thread_sum = 0;
    for(int i=0; i<SpkLiq_threads_N; i++)
    {
        SpkLiq_number_of_inh_neurons_thread_sum += SpkLiq_number_of_inh_neurons_thread[i];
        SpkLiq_number_of_inh_neurons_thread_total[i]=SpkLiq_number_of_inh_neurons_thread_sum;

        SpkLiq_number_of_exc_neurons_thread_sum += SpkLiq_number_of_exc_neurons_thread[i];
        SpkLiq_number_of_exc_neurons_thread_total[i]=SpkLiq_number_of_exc_neurons_thread_sum;

        // Debug
        // printf("Debug - SpkLiq_number_of_inh_neurons_thread_total[%d]:%d\n",i,SpkLiq_number_of_inh_neurons_thread_total[i]);
        // printf("Debug - SpkLiq_number_of_exc_neurons_thread_total[%d]:%d\n",i,SpkLiq_number_of_exc_neurons_thread_total[i]);
    }



    // Spreads the liquid's SpkLiq_inh/exc CONNECTIONS among the threads
    // The connections may not be evenly distributed along the neurons
    for(int i=0; i<SpkLiq_threads_N; i++){

        // Debug
        // printf("Debug - SpkLiq_number_of_inh_neurons_thread[%d]:%d\n",i,SpkLiq_number_of_inh_neurons_thread[i]);
        // printf("Debug - SpkLiq_number_of_exc_neurons_thread[%d]:%d\n",i,SpkLiq_number_of_exc_neurons_thread[i]);

        const int total_inh_final = SpkLiq_neurons_inh_injections_total[SpkLiq_inhibitory_indices[(SpkLiq_number_of_inh_neurons_thread_total[i]-1)]];
        const int total_exc_final = SpkLiq_neurons_exc_injections_total[SpkLiq_excitatory_indices[(SpkLiq_number_of_exc_neurons_thread_total[i]-1)]];

        // Here I define how many connections each thread is going to deal with
        // ATTENTION: I'M REUSING THESE TWO GLOBAL VARIABLES BELOW!!!!
        if(i==0)
        {
          SpkLiq_number_of_inh_connections_thread[i] = total_inh_final;
          SpkLiq_number_of_exc_connections_thread[i] = total_exc_final;
        }else
        {
          const int total_inh_init  = SpkLiq_neurons_inh_injections_total[SpkLiq_inhibitory_indices[(SpkLiq_number_of_inh_neurons_thread_total[i-1]-1)]];
          const int total_exc_init  = SpkLiq_neurons_exc_injections_total[SpkLiq_excitatory_indices[(SpkLiq_number_of_exc_neurons_thread_total[i-1]-1)]];
          SpkLiq_number_of_inh_connections_thread[i] = total_inh_final-total_inh_init;
          SpkLiq_number_of_exc_connections_thread[i] = total_exc_final-total_exc_init;
        }

        // Debug
        // printf("Debug - SpkLiq_number_of_inh_connections_thread[%d]:%d\n",i,SpkLiq_number_of_inh_connections_thread[i]);
        // printf("Debug - SpkLiq_number_of_exc_connections_thread[%d]:%d\n",i,SpkLiq_number_of_exc_connections_thread[i]);
    }



    // These arrays are going to receive the variables addresses from SpikingLiquid_process_internal_spikes_threads_new
    // They work like matrices (array of arrays) being the thread_id the first index.
    SpkLiq_receive_spike_i_thread = malloc(sizeof(int*)*SpkLiq_threads_N);
    SpkLiq_receive_spike_i_w_thread = malloc(sizeof(float*)*SpkLiq_threads_N);
    SpkLiq_receive_spike_i_idx_thread = malloc(sizeof(int)*SpkLiq_threads_N);

    SpkLiq_receive_spike_e_thread = malloc(sizeof(int*)*SpkLiq_threads_N);
    SpkLiq_receive_spike_e_w_thread = malloc(sizeof(float*)*SpkLiq_threads_N);
    SpkLiq_receive_spike_e_idx_thread = malloc(sizeof(int)*SpkLiq_threads_N);

    if (SpkLiq_receive_spike_i_thread==NULL || SpkLiq_receive_spike_i_w_thread==NULL || SpkLiq_receive_spike_i_idx_thread==NULL ||
        SpkLiq_receive_spike_e_thread==NULL || SpkLiq_receive_spike_e_w_thread==NULL || SpkLiq_receive_spike_e_idx_thread==NULL)
    {
        fprintf(stderr, "SpkLiq_receive_spike_ * _thread - malloc error!\n");
        exit(EXIT_FAILURE);
    }


    // These are the arrays each thread are going to use (from the matrices above).
    // Because I'm defining the maximum number of liquid's connections each thread is going to deal with, I can allocate
    // the memory here. According to the liquid's activation level, each thread needs to indicate how much space it used
    // at every update step.
    for(int i=0; i<SpkLiq_threads_N; i++)
    {

        // Debug
        // printf("Debug - Thread [%d] - SpkLiq_number_of_inh_connections_thread[i]:%d \n",i,SpkLiq_number_of_inh_connections_thread[i]);

        SpkLiq_receive_spike_i_thread[i] = malloc(sizeof(int)*SpkLiq_number_of_inh_connections_thread[i]);
        SpkLiq_receive_spike_i_w_thread[i] = malloc(sizeof(float)*SpkLiq_number_of_inh_connections_thread[i]);

        if (SpkLiq_receive_spike_i_thread[i]==NULL || SpkLiq_receive_spike_i_w_thread[i]==NULL){
            fprintf(stderr, "SpkLiq_receive_spike_i*_thread[%d] - malloc error!\n",i);
            exit(EXIT_FAILURE);
        }

        // Debug
        // printf("Debug - Thread [%d] - SpkLiq_number_of_exc_connections_thread[i]:%d \n",i,SpkLiq_number_of_exc_connections_thread[i]);

        SpkLiq_receive_spike_e_thread[i] = malloc(sizeof(int)*SpkLiq_number_of_exc_connections_thread[i]);
        SpkLiq_receive_spike_e_w_thread[i] = malloc(sizeof(float)*SpkLiq_number_of_exc_connections_thread[i]);

        if (SpkLiq_receive_spike_e_thread[i]==NULL || SpkLiq_receive_spike_e_w_thread[i]==NULL){
            fprintf(stderr, "SpkLiq_receive_spike_e*_thread[%d] - malloc error!\n",i);
            exit(EXIT_FAILURE);
        }

    }


    printf("Number of connections INH=>: %d\n", SpkLiq_inh_connections);
    printf("Number of connections EXC=>: %d\n", SpkLiq_exc_connections);
    printf("Total number of connections: %d\n", (SpkLiq_inh_connections+SpkLiq_exc_connections));

    SpkLiq_connected = 1;

}

void SpikingLiquid_Soft_Reset(unsigned int *my_seeds)
{
    //
    // Resets the initial values of the simulation, but keeps the internal structure (connections) of the liquid
    // The use of this function makes possible to restart a simulation without generating again the liquid.
    // Receives an array with the new random seeds.
    //
    // The affected variables are:
    // - float SpkLiq_current_time
    // - int SpkLiq_current_step
    // - unsigned int SpkLiq_rand_r_seeds[5]
    // - rk_state *SpkLiq_threads_states[5]
    // - int *SpkLiq_thread_id
    // - SpkLiq_inh_connections_thread[i]=0
    // - SpkLiq_exc_connections_thread[i]=0
    // - float SpkLiq_neurons_membrane[SpkLiq_number_of_neurons]
    // - float SpkLiq_neurons_membrane_init[SpkLiq_number_of_neurons]
    // - float SpkLiq_neurons_exc_curr[SpkLiq_number_of_neurons]
    // - float SpkLiq_neurons_inh_curr[SpkLiq_number_of_neurons]
    // - float SpkLiq_refrac_timer[SpkLiq_number_of_neurons]
    // - int SpkLiq_test_vthres[SpkLiq_number_of_neurons]
    // - SpkLiq_spike_time[i] = -1.0;//This should indicate that it never spiked
    // - float SpkLiq_noisy_offset_currents[SpkLiq_number_of_neurons]

    printf("Reseting (soft) the simulation...\n");


    // Resets the seeds:
    // RANDOM-0: Membrane initial potentials
    // RANDOM-1: Noisy offset currents
    // RANDOM-2: Selection of the inhibitory and excitatory neurons
    // RANDOM-3: Internal connections of the liquid
    // RANDOM-4: Noisy corrents
    // User random seeds
    for(int i=0;i<5;i++)
    {
        SpkLiq_user_seeds[i] = my_seeds[i];
        printf("Seed SpkLiq_user_seeds[%d]:%u\n",i,SpkLiq_user_seeds[i]);
    }



    // Resets the current time to zero!
    SpkLiq_current_time = 0;
    SpkLiq_current_step = 0;


    //Reseting the array used to save the output spikes
    for(int i=0; i<SpkLiq_number_of_long_ints; i++){
      SpkLiq_test_vthres_bits[i]=0;
    }

    // Resets the random state variables to their initial values
    for(int i=0; i<5; i++)
    {
        SpkLiq_rand_r_seeds[i] = SpkLiq_user_seeds[i];
        // RANDOM-0: Membrane initial potentials
        // RANDOM-1: Noisy offset currents
        // RANDOM-2: Selection of the inhibitory and excitatory neurons
        // RANDOM-3: Internal connections of the liquid
        // RANDOM-4: Noisy corrents

        // Here are generated random seeds for each thread using the user supplied seed as the seed ?!?!? :D
        for(int j=0; j<SpkLiq_threads_N; j++)
        {

            //Initializes (seeds) the random states for each thread (j)
            //In order to start the threads with different seeds, I'm adding "j" to each seed value
            //if the random generator is a good one, this small change would lead to "almost independent" values
            rk_seed(SpkLiq_rand_r_seeds[i]+(unsigned int)j, &SpkLiq_threads_states[i][j]);

            // SpkLiq_threads_seeds[i][j]=(unsigned int)rand_r(&SpkLiq_rand_r_seeds[i]);
            // each row is a random variable (0...4)
            // the columns are the thread indices (0...SpkLiq_threads_N)

            SpkLiq_thread_id[j]=j; // Initializes the thread ids
        }

    }


//
// MULTITHREADS CODE
//

    // Creates the threads to initialize the neuron's variables
    for (int i = 0; i < SpkLiq_threads_N; ++i)
    {
        if (pthread_create(&SpkLiq_threads_ptrs[i], NULL, SpikingLiquid_Reset_thread, &SpkLiq_thread_id[i]))
        {
          fprintf(stderr, "error: SpikingLiquid_Reset_thread\n");
          exit(EXIT_FAILURE);
        }
    }

    // Block (join) all threads - the program is stuck here until all threads return
    for (int i = 0; i < SpkLiq_threads_N; ++i) {
        pthread_join(SpkLiq_threads_ptrs[i], NULL);
    }

}


void SpikingLiquid_Reset()
{
    //
    // Resets the initial values of the simulation, but keeps the internal structure (connections) of the liquid
    // The use of this function makes possible to restart a simulation without generating again the liquid.
    //
    // The affected variables are:
    // - float SpkLiq_current_time
    // - int SpkLiq_current_step
    // - unsigned int SpkLiq_rand_r_seeds[5]
    // - rk_state *SpkLiq_threads_states[5]
    // - int *SpkLiq_thread_id
    // - SpkLiq_inh_connections_thread[i]=0
    // - SpkLiq_exc_connections_thread[i]=0
    // - float SpkLiq_neurons_membrane[SpkLiq_number_of_neurons]
    // - float SpkLiq_neurons_membrane_init[SpkLiq_number_of_neurons]
    // - float SpkLiq_neurons_exc_curr[SpkLiq_number_of_neurons]
    // - float SpkLiq_neurons_inh_curr[SpkLiq_number_of_neurons]
    // - float SpkLiq_refrac_timer[SpkLiq_number_of_neurons]
    // - int SpkLiq_test_vthres[SpkLiq_number_of_neurons]
    // - SpkLiq_spike_time[i] = -1.0;//This should indicate that it never spiked
    // - float SpkLiq_noisy_offset_currents[SpkLiq_number_of_neurons]

    printf("Reseting the simulation...\n");

    // Resets the current time to zero!
    SpkLiq_current_time = 0;
    SpkLiq_current_step = 0;


    //Reseting the array used to save the output spikes
    for(int i=0; i<SpkLiq_number_of_long_ints; i++){
      SpkLiq_test_vthres_bits[i]=0;
    }

    // Resets the random state variables to their initial values
    for(int i=0; i<5; i++)
    {
        SpkLiq_rand_r_seeds[i] = SpkLiq_user_seeds[i];
        // RANDOM-0: Membrane initial potentials / reset voltages
        // RANDOM-1: Noisy offset currents
        // RANDOM-2: Selection of the inhibitory and excitatory neurons
        // RANDOM-3: Internal connections of the liquid
        // RANDOM-4: Noisy corrents

        // Here are generated random seeds for each thread using the user supplied seed as the seed ?!?!? :D
        for(int j=0; j<SpkLiq_threads_N; j++)
        {

            //Initializes (seeds) the random states for each thread (j)
            //In order to start the threads with different seeds, I'm adding "j" to each seed value
            //if the random generator is a good one, this small change would lead to "almost independent" values
            rk_seed(SpkLiq_rand_r_seeds[i]+(unsigned int)j, &SpkLiq_threads_states[i][j]);

            // SpkLiq_threads_seeds[i][j]=(unsigned int)rand_r(&SpkLiq_rand_r_seeds[i]);
            // each row is a random variable (0...4)
            // the columns are the thread indices (0...SpkLiq_threads_N)

            SpkLiq_thread_id[j]=j; // Initializes the thread ids
        }

    }

    for(int i=0; i<SpkLiq_threads_N; i++)
    {
        // Resets the counters used inside the liquid's connections generation
        SpkLiq_inh_connections_thread[i]=0;
        SpkLiq_exc_connections_thread[i]=0;
    }


//
// MULTITHREADS CODE
//

    // Creates the threads to initialize the neuron's variables
    for (int i = 0; i < SpkLiq_threads_N; ++i)
    {
        if (pthread_create(&SpkLiq_threads_ptrs[i], NULL, SpikingLiquid_Reset_thread, &SpkLiq_thread_id[i]))
        {
          fprintf(stderr, "error: SpikingLiquid_Reset_thread\n");
          exit(EXIT_FAILURE);
        }
    }

    // Block (join) all threads - the program is stuck here until all threads return
    for (int i = 0; i < SpkLiq_threads_N; ++i) {
        pthread_join(SpkLiq_threads_ptrs[i], NULL);
    }

}

void *SpikingLiquid_Reset_thread(void *args)
{
    /*
    Resets the neurons internal variables according to the thread id.
    - Each thread must iterate through a slice of the SpkLiq_number_of_neurons and
    uses its own seed value/memory position.

    I've created this as a threaded function because it is necessary to iterate through all the liquid's neurons.

    This functions is called inside the SpikingLiquid_Reset.
    */

    int loop_start, loop_end; //

    int const SpkLiq_tid = *((int *)args); //Receives the number of the thread (thread_id) from the function input arguments

    // Sets up the loop variables according to the thread number
    if (SpkLiq_tid!=(SpkLiq_threads_N-1))
    {
      loop_start = SpkLiq_tid*SpkLiq_number_of_neurons_slice[SpkLiq_tid];
      loop_end = (SpkLiq_number_of_neurons_slice[SpkLiq_tid])*(SpkLiq_tid+1);
    }else
    { // It means the last thread that could have more work to do because of the remainder
      loop_start = SpkLiq_tid*SpkLiq_number_of_neurons_slice[SpkLiq_tid-1]; //Uses the anterior value (the rounded one) and multiplies
      loop_end = (SpkLiq_number_of_neurons_slice[SpkLiq_tid-1])*SpkLiq_tid+(SpkLiq_number_of_neurons_slice[SpkLiq_tid]);
    }


    for(int i=loop_start; i<loop_end; i++)
    {
        // # RANDOM-0
        // # Membrane initial potentials
        // # Initializes the membrane potentials for all neurons randomly according to (Maass, Natschlager, and Markram 2002)
        SpkLiq_neurons_membrane[i] = SpkLiq_membrane_rand[0] + (float)rk_double(&SpkLiq_threads_states[0][SpkLiq_tid]) * (SpkLiq_membrane_rand[1] - SpkLiq_membrane_rand[0]); //in mV
        SpkLiq_neurons_membrane_init[i]=SpkLiq_neurons_membrane[i]; //saves the initial membrane voltage value

        SpkLiq_neurons_exc_curr[i] = 0.0; //Initializes the excitatory currents levels for all neurons

        SpkLiq_neurons_inh_curr[i] = 0.0; //Initializes the inhibitory currents levels for all neurons

        SpkLiq_refrac_timer[i] = 0.0; //Initializes the refractory timers

        SpkLiq_test_vthres[i] = 0; // Initializes all positions with False or NO SPIKES

        SpkLiq_spike_time[i] = -1.0;//This should indicate that it never spiked

        // # RANDOM-1
        // # Noisy offset currents
        // # These are the offset noisy currents according to (Maass, Natschlager, and Markram 2002)
        SpkLiq_noisy_offset_currents[i] = SpkLiq_current_rand[0] + (float)rk_double(&SpkLiq_threads_states[1][SpkLiq_tid]) * (SpkLiq_current_rand[1] - SpkLiq_current_rand[0]); //in nA

        //NEW RANDOM RESETS
        // Initializes the values each neuron membrane assumes after a reset
        SpkLiq_vresets_values[i] = SpkLiq_vresets[0] + (float)rk_double(&SpkLiq_threads_states[1][SpkLiq_tid]) * (SpkLiq_vresets[1] - SpkLiq_vresets[0]); //in mV
    }

    return NULL;
}


void SpikingLiquid_init()
{
    /*

    Initializes the variables and resets everything (calls SpikingLiquid_Reset)

    This functions needs to be called only at the beginning of the simulations.
    After the first run, it's possible to only reset (SpikingLiquid_Reset).

    */

    printf("Generating the liquid internal structure...\n");

    // Calculates the total number of neurons based on the liquid's cuboid shape (x*y*z)
    SpkLiq_number_of_neurons = SpkLiq_net_shape[0]*SpkLiq_net_shape[1]*SpkLiq_net_shape[2];
    SpkLiq_number_of_long_ints = (SpkLiq_number_of_neurons>>6)+((SpkLiq_number_of_neurons&(64-1))>0); //the same as: (SpkLiq_number_of_neurons/64 + (SpkLiq_number_of_neurons%64>0))


    // This is going to be used to control the memory positions each thread can access.
    int SpkLiq_number_of_neurons_in_slice_init = SpkLiq_number_of_neurons/SpkLiq_threads_N;
    if (SpkLiq_number_of_neurons_in_slice_init<2)
    {
      fprintf(stderr,"SpikingLiquid_init: SpkLiq_number_of_neurons_in_slice_init - too many threads (too few neurons)!\n");
      exit(EXIT_FAILURE);
    }

    SpkLiq_number_of_neurons_slice = malloc(sizeof(int)*SpkLiq_number_of_neurons_in_slice_init);
    if (SpkLiq_number_of_neurons_slice==NULL)
    {
      fprintf(stderr,"SpikingLiquid_init: SpkLiq_number_of_neurons_slice - malloc ERROR!");
      exit(EXIT_FAILURE);
    }

    // Spreads the liquid's indices among the threads
    for(int i=0; i<SpkLiq_threads_N; i++)
    {
        SpkLiq_number_of_neurons_slice[i] = SpkLiq_number_of_neurons_in_slice_init;
    }

    // Adjusts the amount of "work" each thread should get if the division is not an integer
    if (SpkLiq_number_of_neurons % SpkLiq_threads_N != 0)
    { // Verifies if there's a remainder
        // In the case of a remainder, makes the last thread "heavier" adding the remainder
        SpkLiq_number_of_neurons_slice[SpkLiq_threads_N-1] = SpkLiq_number_of_neurons_in_slice_init + (SpkLiq_number_of_neurons % SpkLiq_threads_N);
    }



    // Calculates the number of inhibitory and excitatory neurons
    SpkLiq_number_of_inh_neurons = (int) (SpkLiq_number_of_neurons*(SpkLiq_inhibitory_percentage/100.0)); // The int casting is truncating into an integer
    SpkLiq_number_of_exc_neurons = SpkLiq_number_of_neurons-SpkLiq_number_of_inh_neurons;


    //
    // Memory allocations
    //

    // Here I need to allocate memory for all the dynamic variables acessed by SpikingLiquid_Reset.
    SpkLiq_neurons_membrane = malloc(sizeof(float)*SpkLiq_number_of_neurons);
    SpkLiq_neurons_membrane_init = malloc(sizeof(float)*SpkLiq_number_of_neurons);
    SpkLiq_neurons_exc_curr = malloc(sizeof(float)*SpkLiq_number_of_neurons);
    SpkLiq_neurons_inh_curr = malloc(sizeof(float)*SpkLiq_number_of_neurons);
    SpkLiq_refrac_timer = malloc(sizeof(float)*SpkLiq_number_of_neurons);
    SpkLiq_test_vthres = malloc(sizeof(int)*SpkLiq_number_of_neurons);
    SpkLiq_noisy_offset_currents = malloc(sizeof(float)*SpkLiq_number_of_neurons);
    SpkLiq_noisy_currents = malloc(sizeof(float)*SpkLiq_number_of_neurons);

    //NEW RANDOM RESETS
    SpkLiq_vresets_values = malloc(sizeof(float)*SpkLiq_number_of_neurons);

    // And also the variables used in this function, but that are not going to be reset by SpikingLiquid_Reset
    SpkLiq_inhibitory_indices = malloc(sizeof(int)*SpkLiq_number_of_inh_neurons);
    SpkLiq_excitatory_indices = malloc(sizeof(int)*SpkLiq_number_of_exc_neurons);

    SpkLiq_liquid_indices_shuffled = malloc(sizeof(int)*SpkLiq_number_of_neurons);

    SpkLiq_refrac_values = malloc(sizeof(float)*SpkLiq_number_of_neurons);

    // Used inside the SpikingLiquid_generate_connections_less_memory_thread
    SpkLiq_pre_i_thread = malloc(sizeof(int*)*SpkLiq_threads_N);
    SpkLiq_pos_i_thread = malloc(sizeof(int*)*SpkLiq_threads_N);
    SpkLiq_w_i_thread = malloc(sizeof(float*)*SpkLiq_threads_N);

    SpkLiq_pre_e_thread = malloc(sizeof(int*)*SpkLiq_threads_N);
    SpkLiq_pos_e_thread = malloc(sizeof(int*)*SpkLiq_threads_N);
    SpkLiq_w_e_thread = malloc(sizeof(float*)*SpkLiq_threads_N);


    // Actually they are reset to zero inside the SpikingLiquid_Reset
    // They are used as counters during the generation of the liquid's connections
    SpkLiq_inh_connections_thread = malloc(sizeof(int)*SpkLiq_threads_N);
    SpkLiq_exc_connections_thread = malloc(sizeof(int)*SpkLiq_threads_N);

    // These are used during the update of the liquid to keep the number of connections each thread will process
    SpkLiq_number_of_inh_connections_thread = malloc(sizeof(int)*SpkLiq_threads_N);
    SpkLiq_number_of_exc_connections_thread = malloc(sizeof(int)*SpkLiq_threads_N);

    SpkLiq_3Didx = malloc(sizeof(Array3D)*SpkLiq_number_of_neurons); //stores the (x,y,z) positions of each neuron in the liquid

    SpkLiq_spike_time = malloc(sizeof(float)*SpkLiq_number_of_neurons); //stores the time of each neuron's last spike

    SpkLiq_neurons_connected = calloc(SpkLiq_number_of_neurons,sizeof(int));

    // Tests to verify if the malloc was successful with all the variables
    if (SpkLiq_neurons_membrane==NULL || SpkLiq_neurons_exc_curr==NULL || SpkLiq_neurons_inh_curr==NULL ||
        SpkLiq_refrac_timer==NULL || SpkLiq_test_vthres==NULL ||
        SpkLiq_noisy_offset_currents==NULL || SpkLiq_noisy_currents==NULL || SpkLiq_inhibitory_indices==NULL || SpkLiq_excitatory_indices==NULL ||
        SpkLiq_liquid_indices_shuffled==NULL || SpkLiq_refrac_values==NULL ||
        SpkLiq_pre_i_thread==NULL || SpkLiq_pos_i_thread==NULL || SpkLiq_w_i_thread==NULL ||
        SpkLiq_pre_e_thread==NULL || SpkLiq_pos_e_thread==NULL || SpkLiq_w_e_thread==NULL ||
        SpkLiq_inh_connections_thread==NULL || SpkLiq_exc_connections_thread==NULL || SpkLiq_3Didx==NULL || SpkLiq_spike_time==NULL || SpkLiq_neurons_connected==NULL){

        fprintf(stderr,"SpikingLiquid_init - Malloc ERROR!");
        exit(EXIT_FAILURE); //I need to check which error I should signal here...
    }

    //
    // These memory allocations should be moved together with the ones above to save some ifs...
    //

    //I think it's easier to waste memomy instead of having a smaller array the size of number of excitatory/inhibitory neurons
    SpkLiq_neurons_exc_injections = malloc(sizeof(int *)*SpkLiq_number_of_neurons);
    SpkLiq_neurons_exc_injections_total = calloc(SpkLiq_number_of_neurons,sizeof(int));
    SpkLiq_neurons_inh_injections = malloc(sizeof(int *)*SpkLiq_number_of_neurons);
    SpkLiq_neurons_inh_injections_total = calloc(SpkLiq_number_of_neurons,sizeof(int));
    SpkLiq_neurons_exc_injections_w = malloc(sizeof(float *)*SpkLiq_number_of_neurons);
    SpkLiq_neurons_inh_injections_w = malloc(sizeof(float *)*SpkLiq_number_of_neurons);

    // Allocates the number of bytes necessary to store all the spikes (and writes a zero into each one).
    SpkLiq_test_vthres_bits = calloc(SpkLiq_number_of_long_ints,sizeof(long int));
    if (SpkLiq_test_vthres_bits==NULL){
        fprintf(stderr,"SpkLiq_test_vthres_bits - columns- malloc ERROR!");
        exit(EXIT_FAILURE);
    }

    // Allocation of the multiple threads random seeds
    // Each row represents one of the 5 random variables (Random-0...4)
    // Then the columns are the threads indices.
    for (int i=0; i<5; i++){
        SpkLiq_threads_seeds[i] = malloc(sizeof(unsigned int)*SpkLiq_threads_N);
        if (SpkLiq_threads_seeds[i]==NULL){
            fprintf(stderr,"SpkLiq_threads_seeds - columns- malloc ERROR!");
            exit(EXIT_FAILURE);
        }

        // These are the random states used by the rk_* functions
        SpkLiq_threads_states[i] = malloc(sizeof(rk_state)*SpkLiq_threads_N);
        if (SpkLiq_threads_states[i]==NULL){
            fprintf(stderr,"SpkLiq_threads_states - columns- malloc ERROR!");
            exit(EXIT_FAILURE);
        }

    }


    // Allocates the memory positions to be used with the threads call
    SpkLiq_threads_ptrs = malloc(sizeof(pthread_t)*SpkLiq_threads_N);
    if (SpkLiq_threads_ptrs==NULL){
        fprintf(stderr,"SpkLiq_threads_ptrs - malloc ERROR!");
        exit(EXIT_FAILURE);
    }

    // Allocates the memory positions to be used with the threads ids
    SpkLiq_thread_id = malloc(sizeof(int)*SpkLiq_threads_N);
    if (SpkLiq_thread_id==NULL){
        fprintf(stderr,"SpkLiq_thread_id - malloc ERROR!");
        exit(EXIT_FAILURE);
    }


    SpkLiq_lbd_value_opt = 1/(SpkLiq_lbd_value*SpkLiq_lbd_value); // This value is used a lot of times, so it's better to
                                                                  // do the division only once.
                                                                  // (see euler_distance_opt)

    printf("Calling SpikingLiquid_Reset from the SpikingLiquid_init...\n");

    // Initializes (resets) the internal variables to start the simulation
    // But keeps the structure (not in this case because there is no structure yet...)
    SpikingLiquid_Reset();

    printf("Total number of neurons: %d\n", SpkLiq_number_of_neurons);
    printf("Total number of INH neurons: %d\n", SpkLiq_number_of_inh_neurons);
    printf("Total number of EXC neurons: %d\n", SpkLiq_number_of_exc_neurons);

    SpkLiq_initialized = 1;
    SpkLiq_connected = 0; //resets the flag, so it's going to be up only after the processing of the connections
}


void SpkLiq_process_exc_spikes(const int *restrict const spikes, const float *restrict const weights, int const number_of_spikes)
{
    /*
    Processes the received spikes at the current time
    spikes: list with the indexes of the neurons who spiked.
    weights: the weights to the neurons who spiked

    THIS IS NOT THREAD SAFE BECAUSE A NEURON CAN RECEIVE MORE THAN ONE SPIKE AT EACH TIME STEP
    */

    float *restrict const SpkLiq_neurons_exc_curr_internal = SpkLiq_neurons_exc_curr; // Trying to help the compiler to optimize...

    for(int i=0; i<number_of_spikes; i++){
        SpkLiq_neurons_exc_curr_internal[spikes[i]] +=  weights[i]; // Adds to the ones that received spikes
        // Debug
        // printf("Debug - SpkLiq_process_exc_spikes - spikes[i]:%d\n", spikes[i]);
    }
}


void SpkLiq_process_inh_spikes(const int *restrict const spikes, const float *restrict const weights, int const number_of_spikes)
{
    /*
    Processes the received spikes at the current time
    spikes: list with the indexes of the neurons who spiked.
    weights: the weights to the neurons who spiked

    THIS IS NOT THREAD SAFE BECAUSE A NEURON CAN RECEIVE MORE THAN ONE SPIKE AT EACH TIME STEP
    */

    float *restrict const SpkLiq_neurons_inh_curr_internal = SpkLiq_neurons_inh_curr; // Trying to help the compiler to optimize...

    for(int i=0; i<number_of_spikes; i++){
        SpkLiq_neurons_inh_curr_internal[spikes[i]] +=  weights[i]; // Adds to the ones that received spikes
        // Debug
        // printf("Debug - SpkLiq_process_inh_spikes - spikes[i]:%d\n", spikes[i]);
    }
}


void *SpikingLiquid_process_internal_spikes_threads_new(void *args)
{
    /*
    Tests if the internal neurons spiked
    This function is called every simulation step, so it's important to optimize for speed!

    The const ... restrict const stuff is my attempt to give hints to gcc and help with compiler optimization!
    */

    const int SpkLiq_tid = *((int *)args); //Receives the number of the thread (thread_id) from the function input arguments

    const int *restrict const SpkLiq_test_vthres_internal = SpkLiq_test_vthres; // array with something valued as True if the neuron spiked in the last step
    const int *restrict const SpkLiq_inhibitory_indices_internal = SpkLiq_inhibitory_indices; // array with an ascending ordered list of the inhibitory neuron indices
    const int *restrict const SpkLiq_excitatory_indices_internal = SpkLiq_excitatory_indices; // array with an ascending ordered list of the excitatory neuron indices
    const int **restrict const SpkLiq_neurons_inh_injections_internal = (const int **)SpkLiq_neurons_inh_injections; // matrix with all POST the connections each inhibitory neuron has
    const float **restrict const SpkLiq_neurons_inh_injections_w_internal = (const float **)SpkLiq_neurons_inh_injections_w; // the weights from the connections above
    const int **restrict const SpkLiq_neurons_exc_injections_internal = (const int **)SpkLiq_neurons_exc_injections; // matrix with all POST the connections each excitatory neuron has
    const float **restrict const SpkLiq_neurons_exc_injections_w_internal = (const float **)SpkLiq_neurons_exc_injections_w; // the weights from the connections above
                                                                            // the type cast is to make gcc stop complaining about something that looks correct...

    int **restrict const SpkLiq_receive_spike_i_thread_internal = SpkLiq_receive_spike_i_thread;
    float **restrict const SpkLiq_receive_spike_i_w_thread_internal = SpkLiq_receive_spike_i_w_thread;

    int **restrict const SpkLiq_receive_spike_e_thread_internal = SpkLiq_receive_spike_e_thread;
    float **restrict const SpkLiq_receive_spike_e_w_thread_internal = SpkLiq_receive_spike_e_w_thread;


    int loop_inh_start=0,loop_exc_start=0;
    const int loop_inh_end = (SpkLiq_number_of_inh_neurons_thread_total[SpkLiq_tid]);
    const int loop_exc_end = (SpkLiq_number_of_exc_neurons_thread_total[SpkLiq_tid]);

    if(SpkLiq_tid>0)
    {
      loop_inh_start  = (SpkLiq_number_of_inh_neurons_thread_total[SpkLiq_tid-1]);
      loop_exc_start  = (SpkLiq_number_of_exc_neurons_thread_total[SpkLiq_tid-1]);
    }


    //
    // PROCESS THE INHIBITORY => ? CONNECTIONS
    //

    // Debug
    // printf("Debug - Thread [%d] - (INH)loop_inh_start:%d | loop_inh_end:%d\n",SpkLiq_tid,loop_inh_start,loop_inh_end);

    int spike_i_idx = 0; // counts how many INHIBITORY spikes were generated
    for(int i=loop_inh_start; i<loop_inh_end; i++)
    {
      const int inh_idx = SpkLiq_inhibitory_indices_internal[i];

      if(SpkLiq_test_vthres_internal[inh_idx]) // goes through only the INHIBITORY neurons to check for spikes
      {
          for(int j=0; j<SpkLiq_neurons_inh_injections_internal[inh_idx][0]; j++) // The first item '[0]' has the total number of POST connections
          {
            // saves a list of indices and weights to be used with SpkLiq_process_inh_spikes
            SpkLiq_receive_spike_i_thread_internal[SpkLiq_tid][spike_i_idx]=SpkLiq_neurons_inh_injections_internal[inh_idx][j+1];
            SpkLiq_receive_spike_i_w_thread_internal[SpkLiq_tid][spike_i_idx]=SpkLiq_neurons_inh_injections_w_internal[inh_idx][j+1];
            spike_i_idx++;
          }
      }
    }
    SpkLiq_receive_spike_i_idx_thread[SpkLiq_tid]=spike_i_idx; // stores the total number of inhibitory spikes to be used
                                                               // with the SpkLiq_process_inh_spikes

    //
    // PROCESS THE EXCITATORY => ? CONNECTIONS
    //

    // Debug
    // printf("Debug - Thread [%d] - (EXC)loop_exc_start:%d | loop_exc_end:%d\n",SpkLiq_tid,loop_exc_start,loop_exc_end);

    int spike_e_idx = 0; // counts how many EXCITATORY spikes were generated
    for(int i=loop_exc_start; i<loop_exc_end; i++)
    {
      const int exc_idx = SpkLiq_excitatory_indices_internal[i];
      if(SpkLiq_test_vthres_internal[exc_idx])
      {
          for(int j=0; j<SpkLiq_neurons_exc_injections_internal[exc_idx][0]; j++)
          {
            SpkLiq_receive_spike_e_thread_internal[SpkLiq_tid][spike_e_idx]=SpkLiq_neurons_exc_injections_internal[exc_idx][j+1];
            SpkLiq_receive_spike_e_w_thread_internal[SpkLiq_tid][spike_e_idx]=SpkLiq_neurons_exc_injections_w_internal[exc_idx][j+1];
            spike_e_idx++;
          }
      }
    }
    SpkLiq_receive_spike_e_idx_thread[SpkLiq_tid]=spike_e_idx;

    // Debug
    // printf("Debug - SpikingLiquid_process_internal_spikes_threads_new[%d]\n",SpkLiq_tid);


    return NULL;
}



void *SpikingLiquid_update_internal_thread(void *args)
{
  /*
    Updates the values of the membrane voltage after ALL the currents had been processed.
  */

    const int SpkLiq_tid = *((int *)args); //Receives the number of the thread (thread_id) from the function input arguments

    float *restrict const SpkLiq_vresets_internal = SpkLiq_vresets_values;
    const float SpkLiq_noisy_current_rand_internal = SpkLiq_noisy_current_rand;
    const float SpkLiq_step_internal = SpkLiq_step;
    const float SpkLiq_taum_internal = SpkLiq_taum;
    const float SpkLiq_current_time_internal = SpkLiq_current_time;
    const float SpkLiq_vrest_internal = SpkLiq_vrest;
    const float SpkLiq_cm_internal = SpkLiq_cm;

    rk_state *restrict const SpkLiq_threads_states_internal = &SpkLiq_threads_states[4][SpkLiq_tid];

    float *restrict const SpkLiq_noisy_currents_internal = SpkLiq_noisy_currents;
    float *restrict const SpkLiq_refrac_timer_internal = SpkLiq_refrac_timer;

    float *restrict const SpkLiq_neurons_membrane_internal = SpkLiq_neurons_membrane;
    float *restrict const SpkLiq_spike_time_internal = SpkLiq_spike_time;

    const float *restrict const SpkLiq_noisy_offset_currents_internal = SpkLiq_noisy_offset_currents;
    const float *restrict const SpkLiq_neurons_inh_curr_internal = SpkLiq_neurons_inh_curr;
    const float *restrict const SpkLiq_neurons_exc_curr_internal = SpkLiq_neurons_exc_curr;

    int *restrict const SpkLiq_test_vthres_internal = SpkLiq_test_vthres;

    int loop_start, loop_end;

    // Sets up the loop variables according to the thread number
    if (SpkLiq_tid!=(SpkLiq_threads_N-1))
    {
      loop_start = SpkLiq_tid*SpkLiq_number_of_neurons_slice[SpkLiq_tid];
      loop_end = (SpkLiq_number_of_neurons_slice[SpkLiq_tid])*(SpkLiq_tid+1);
    }else
    { // It means the last thread that could have more work to do because of the remainder
      loop_start = SpkLiq_tid*SpkLiq_number_of_neurons_slice[SpkLiq_tid-1]; //Uses the anterior value (the rounded one) and multiplies
      loop_end = (SpkLiq_number_of_neurons_slice[SpkLiq_tid-1])*SpkLiq_tid+(SpkLiq_number_of_neurons_slice[SpkLiq_tid]);
    }


    for(int i=(const int)loop_start; i<(const int)loop_end; i++)
    {
      if(SpkLiq_neurons_connected[i])
      //if(1)
      { //Only updates neurons that are actually connected to another ones!

        //Check if the neuron spiked by comparing its the membrane voltage with the threshold
        if(SpkLiq_neurons_membrane_internal[i]>=SpkLiq_vthres)
        {
            // SpkLiq_test_vthres_internal[i] = 1; //# Verifies who should spike and creates a boolean vector with this information
            SpkLiq_test_vthres_internal[i] = i+1; //Saves the index of the neuron who spiked (from 1 to SpkLiq_number_of_neurons)
                                                  //Using the index instead of 0 and 1 makes easier to process this array because
                                                  //zero is not related anymore to a neuron's index!
/////FAST SET BIT ON
            SpkLiq_test_vthres_bits[i>>6] |= (1<<(i&(64-1)));

            SpkLiq_neurons_membrane_internal[i] = SpkLiq_vresets_internal[i]; //# Resets the membrane of the spiked ones
            SpkLiq_refrac_timer_internal[i] = SpkLiq_refrac_values[i]; //# Resets the refractory timer to the ones who spiked

            SpkLiq_spike_time_internal[i] = SpkLiq_current_time_internal; //Indicates the time the neuron spiked
        }else
        {
            SpkLiq_test_vthres_internal[i] = 0; //# Verifies who should spike and creates a boolean vector with this information
/////FAST SET BIT OFF
            SpkLiq_test_vthres_bits[i>>6] &= ~(1<<(i&(64-1)));
        }


        // Generates the new values to the noisy currents
        // # RANDOM-4
        // # Noisy corrents
        SpkLiq_noisy_currents_internal[i] = (float)rk_normal(SpkLiq_threads_states_internal,0,1)*SpkLiq_noisy_current_rand_internal;

        // # After the currents, it is necessary to update the membrane voltage
        // # Integrating the function:
        // dv/dt  = (ie + ii + i_offset + i_inj)/c_m + (v_rest-v)/tau_m
        if(SpkLiq_refrac_timer_internal[i]>0)
        {
            SpkLiq_neurons_membrane_internal[i] = SpkLiq_vresets_internal[i]; // Resets the membrane of the refractory ones
        }else
        { //If the membrane did not reset, then it is necessary to integrate
            SpkLiq_neurons_membrane_internal[i] += ((SpkLiq_neurons_exc_curr_internal[i] + SpkLiq_neurons_inh_curr_internal[i] + SpkLiq_noisy_offset_currents_internal[i] + SpkLiq_noisy_currents_internal[i])/SpkLiq_cm_internal + (SpkLiq_vrest_internal-SpkLiq_neurons_membrane_internal[i])/SpkLiq_taum_internal)*SpkLiq_step_internal;
        }

//////HERE I'M CLAMPING THE POSITIVE VALUE OF THE MEMBRANE VOLTAGE
        if(SpkLiq_neurons_membrane_internal[i]>SpkLiq_vthres)
            SpkLiq_neurons_membrane_internal[i]=SpkLiq_vthres; //Limits the maximum value


//////HERE I'M SUPPOSING THE SIMULATOR IS ALWAYS GOING TO USE POSITIVE VALUES TO SpkLiq_vthres
        if(SpkLiq_neurons_membrane_internal[i]<-SpkLiq_vthres)
            SpkLiq_neurons_membrane_internal[i]=-SpkLiq_vthres; //Limits the minimum value


        if (SpkLiq_refrac_timer_internal[i]<0)
                  SpkLiq_refrac_timer_internal[i]=0;

        SpkLiq_refrac_timer_internal[i] -= SpkLiq_step_internal; //# Subtracts the refractory timer to enable the refractory calculation
      }
    }
    return NULL;
}


void SpikingLiquid_update(const int *restrict const spikes_exc, const int *restrict const spikes_inh, const float *restrict const weights_exc, const float *restrict const weights_inh, int const size_exc, int const size_inh)
{
    /*
    Updates the simulation, running one time SpkLiq_step, and processes the received excitatory and inhibitory spikes.

    SpikingLiquid_update(spikes_exc, spikes_inh, weights_exc, weights_inh):
    spikes_exc: list/array with the index of the neurons receiving excitatory spikes
    spikes_inh: list/array with the index of the neurons receiving inhibitory spikes
    weights_exc: list/array with the weights in the same order as spikes_exc
    weights_inh: list/array with the weights in the same order as spikes_inh

    The first update is time=0s

    I need to verify if these functions are thread safe:
        SpikingLiquid_process_internal_spikes
        SpkLiq_process_inh_spikes
        SpkLiq_process_exc_spikes

    In order to optimize this function, I think it needs a static variable to control
    the threads initialization, so the overhead occurs only during the first call.

    */

    float const SpkLiq_taui_internal = SpkLiq_taui;
    float const SpkLiq_taue_internal = SpkLiq_taue;
    float const SpkLiq_step_internal = SpkLiq_step;

    float *restrict SpkLiq_neurons_inh_curr_internal = SpkLiq_neurons_inh_curr;
    float *restrict SpkLiq_neurons_exc_curr_internal = SpkLiq_neurons_exc_curr;

// THIS IS FAST ENOUGH
    // Solves the diff equation for the currents ( die(t)/dt=-ie(t)/taue and dii(t)/dt=-ii(t)/taui )
    // The membrane depends on these two currents, so they must be solved before the membrane.
    // These updates also must occur BEFORE the processing of the spikes, otherwise the spikes are going
    // to be reduced just like when one time SpkLiq_step had passed.
    for(int i=0; i<SpkLiq_number_of_neurons; i++){
///////////////////////////////////////////////////////////////////////////
      if(SpkLiq_neurons_connected[i])
      //if(1)
      { //Only updates neurons that are actually connected to another ones!
        SpkLiq_neurons_inh_curr_internal[i] += (-SpkLiq_neurons_inh_curr_internal[i]/SpkLiq_taui_internal)*SpkLiq_step_internal;
        // Updates the INHIBITORY current values
        // integrating the function dii/dt = -ii/SpkLiq_taui
        SpkLiq_neurons_exc_curr_internal[i] += (-SpkLiq_neurons_exc_curr_internal[i]/SpkLiq_taue_internal)*SpkLiq_step_internal;
        // Updates the EXCITATORY current values
        // integrating the function die/dt = -ie/taue
      }
    }


//
// MULTITHREADS CODE
// - This piece of code is the one that spends almost all the time during the update, but it scales up very nicelly.
//
//

    // Creates the threads to processes the internal spikes
    for (int i=0; i < SpkLiq_threads_N; ++i) {
        if (pthread_create(&SpkLiq_threads_ptrs[i], NULL, SpikingLiquid_process_internal_spikes_threads_new, &SpkLiq_thread_id[i])) {
          fprintf(stderr, "error: SpikingLiquid_process_internal_spikes_threads\n");
          exit(EXIT_FAILURE);
        }
    }
    // Block (join) all threads - the program is stuck here until all threads return
    for (int i=0; i < SpkLiq_threads_N; ++i) {
        pthread_join(SpkLiq_threads_ptrs[i], NULL);
    }


///// THE PROBLEM IS OCCURING HERE!!!!
///// When I change both SpkLiq_receive_spike_i/e_idx_thread to ZERO it works.

// SERIAL
    // Processes the information generated by the threads
    // This cannot be done in parallel because some neurons can receive more than one spike and
    // then the threads could try to update the same synapse current at the same time!
    for(int i=0; i<SpkLiq_threads_N; i++){
        SpkLiq_process_inh_spikes(SpkLiq_receive_spike_i_thread[i], SpkLiq_receive_spike_i_w_thread[i], SpkLiq_receive_spike_i_idx_thread[i]);
        SpkLiq_process_exc_spikes(SpkLiq_receive_spike_e_thread[i], SpkLiq_receive_spike_e_w_thread[i], SpkLiq_receive_spike_e_idx_thread[i]);
    }


// SERIAL
    // Processes the EXTERNAL spikes (inhibitory and excitatory)
    if(size_inh!=0)
        SpkLiq_process_inh_spikes(spikes_inh, weights_inh, size_inh);

    if(size_exc!=0)
        SpkLiq_process_exc_spikes(spikes_exc, weights_exc, size_exc);

//
// MULTITHREADS CODE
//

    // Creates the threads to processes the membrane, noisy currents, etc
    for(int i=0; i < SpkLiq_threads_N; ++i) {
        if (pthread_create(&SpkLiq_threads_ptrs[i], NULL, SpikingLiquid_update_internal_thread, &SpkLiq_thread_id[i])) {
          fprintf(stderr, "error: SpikingLiquid_update_internal_thread\n");
          exit(EXIT_FAILURE);
        }
    }
    // Block (join) all threads - the program is stuck here until all threads return
    for(int i=0; i < SpkLiq_threads_N; ++i) {
        pthread_join(SpkLiq_threads_ptrs[i], NULL);
    }


    //# If I update the time here makes it possible to generate things at time zero!
    SpkLiq_current_time += SpkLiq_step_internal; //# Advances the simulation time one SpkLiq_step
    SpkLiq_current_step++; // Advances the simulation time one step
}


// ##########################################################################################
// ##########################################################################################
// ###################################USER INTERFACE#########################################
// ##########################################################################################
// ##########################################################################################

// If the user wants to create connections, it's necessary to populate:
// - int *SpkLiq_inhibitory_indices: array with the neuron's indices belonging to inhibitory group
// - int *SpkLiq_excitatory_indices: array with the neuron's indices belonging to excitatory group
// - int SpkLiq_inh_connections: total number of inhibitory=>? connections
// - int SpkLiq_exc_connections: total number of excitatory=>? connections

// - int *SpkLiq_pre_i: stores the indices of the inhibitory=>? PRE-synaptic connections in the liquid
// - int *SpkLiq_pos_i: stores the indices of the inhibitory=>? POS-synaptic connections in the liquid
// - float *SpkLiq_w_i: stores the weights of the inhibitory connections above

// - int *SpkLiq_pre_e: stores the indices of the excitatory=>? PRE-synaptic connections in the liquid
// - int *SpkLiq_pos_e: stores the indices of the excitatory=>? POS-synaptic connections in the liquid
// - float *SpkLiq_w_e: stores the weights of the excitatory connections above

// - int *restrict SpkLiq_neurons_connected; //Signalize that this neuron has at least ONE connection to other neuron

int user_setup(
                int *_SpkLiq_net_shape,
                float _SpkLiq_lbd_value,
                float _SpkLiq_step,
                float _SpkLiq_taum,
                float _SpkLiq_cm,
                float _SpkLiq_taue,
                float _SpkLiq_taui,
                float *_SpkLiq_membrane_rand,
                float *_SpkLiq_current_rand,
                float _SpkLiq_noisy_current_rand,
                float *_SpkLiq_vresets,
                float _SpkLiq_vthres,
                float _SpkLiq_vrest,
                float *_SpkLiq_refractory,
                float _SpkLiq_inhibitory_percentage,
                float _SpkLiq_min_perc,
                unsigned int *my_seeds,
                int _SpkLiq_threads_N
                ){
    //
    // USER PARAMETERS
    //

    // Network user parameters
    SpkLiq_net_shape[0] = _SpkLiq_net_shape[0];//15;
    SpkLiq_net_shape[1] = _SpkLiq_net_shape[1];//30;
    SpkLiq_net_shape[2] = _SpkLiq_net_shape[2];//3;

    SpkLiq_lbd_value = _SpkLiq_lbd_value;//1.2;
    SpkLiq_step = _SpkLiq_step;//0.2E-3;
    SpkLiq_taum = _SpkLiq_taum;//30E-3;
    SpkLiq_cm = _SpkLiq_cm;//30E-9;
    SpkLiq_taue = _SpkLiq_taue;//3E-3;
    SpkLiq_taui = _SpkLiq_taui;//6E-3;

    SpkLiq_membrane_rand[0] = _SpkLiq_membrane_rand[0];//13.5E-3;
    SpkLiq_membrane_rand[1] = _SpkLiq_membrane_rand[1];//15E-3; //range of the possible values to the initialization of the membrane voltage


    SpkLiq_current_rand[0] = _SpkLiq_current_rand[0];//14.975E-9;
    SpkLiq_current_rand[1] = _SpkLiq_current_rand[1];//15.025E-9; //range of the possible values to the initialization of the constant noisy offset current

    SpkLiq_noisy_current_rand = _SpkLiq_noisy_current_rand;//0.2E-9; //max range of the possible values to the initialization of the variable noisy current

    SpkLiq_vresets[0] = _SpkLiq_vresets[0];
    SpkLiq_vresets[1] = _SpkLiq_vresets[1];

    SpkLiq_vthres = _SpkLiq_vthres;//15E-3;
    SpkLiq_vrest = _SpkLiq_vrest;//0.0;

    SpkLiq_refractory[0] = _SpkLiq_refractory[0];//3E-3;
    SpkLiq_refractory[1] = _SpkLiq_refractory[1];//2E-3;
    SpkLiq_inhibitory_percentage = _SpkLiq_inhibitory_percentage;//20; //it means the liquid is going to have approximately 20% of inhibitory neurons

    SpkLiq_min_perc = _SpkLiq_min_perc; //mininum percentage used with the distance based connection generation

    // RANDOM-0: Membrane initial potentials
    // RANDOM-1: Noisy offset currents
    // RANDOM-2: Selection of the inhibitory and excitatory neurons
    // RANDOM-3: Internal connections of the liquid
    // RANDOM-4: Noisy corrents
    // User random seeds
    for(int i=0;i<5;i++)
        SpkLiq_user_seeds[i] = my_seeds[i];

    SpkLiq_threads_N = _SpkLiq_threads_N;

    return EXIT_SUCCESS;
}


int reads_spikes(int *output){
  // Returns only the indices of the neurons who spiked
  // The indices are returned using the output pointer and
  // the total number of spikes is returned by the function.
  int output_idx = 0;
  for(int i=0; i<SpkLiq_number_of_neurons; i++){
    if (SpkLiq_test_vthres[i]){
      output[output_idx] = i;
      output_idx++;
    }
  }
  return output_idx; //returns the total number of spikes
}

void reads_test_vthres_bits(int *output){
    memcpy(output,SpkLiq_test_vthres_bits,sizeof(long int)*SpkLiq_number_of_long_ints); // spikes output from the liquid
}

void reads_test_vthres(int *output){
    memcpy(output,SpkLiq_test_vthres,sizeof(int)*SpkLiq_number_of_neurons); // spikes output from the liquid
}

void writes_membranes(float *output){
    memcpy(SpkLiq_neurons_membrane,output,sizeof(float)*SpkLiq_number_of_neurons); // overwrites the membranes voltages
}

void reads_membranes(float *output){
    memcpy(output,SpkLiq_neurons_membrane,sizeof(float)*SpkLiq_number_of_neurons); // membranes voltages
}

void reads_membranes_init(float *output){
    memcpy(output,SpkLiq_neurons_membrane_init,sizeof(float)*SpkLiq_number_of_neurons); // initial membranes voltages
}

void writes_membranes_init(float *output){
    memcpy(SpkLiq_neurons_membrane_init,output,sizeof(float)*SpkLiq_number_of_neurons); // initial membranes voltages
}

void reads_exc_synapses(float *output){
    memcpy(output,SpkLiq_neurons_exc_curr,sizeof(float)*SpkLiq_number_of_neurons); // excitatory currents
}

void reads_inh_synapses(float *output){
    memcpy(output,SpkLiq_neurons_inh_curr,sizeof(float)*SpkLiq_number_of_neurons); // inhibitory currents
}

// NEW REALLOC
void writes_pre_i_connections(int *output, int number_of_connections){
  SpkLiq_pre_i = realloc(SpkLiq_pre_i,sizeof(int)*number_of_connections);
  if (SpkLiq_pre_i==NULL)
  {
    fprintf(stderr,"SpkLiq_pre_i - realloc error\n");
    exit(EXIT_FAILURE);
  }
  memcpy(SpkLiq_pre_i,output,sizeof(int)*number_of_connections); // overwrites the inhibitory PRE connections
}

void reads_pre_i_connections(int *output){
    memcpy(output,SpkLiq_pre_i,sizeof(int)*SpkLiq_inh_connections);
}

// NEW REALLOC
void writes_pos_i_connections(int *output, int number_of_connections){
  SpkLiq_pos_i = realloc(SpkLiq_pos_i,sizeof(int)*number_of_connections);
  if (SpkLiq_pos_i==NULL)
  {
    fprintf(stderr,"SpkLiq_pos_i - realloc error\n");
    exit(EXIT_FAILURE);
  }
  memcpy(SpkLiq_pos_i,output,sizeof(int)*number_of_connections); // overwrites the inhibitory PRE connections
}

void reads_pos_i_connections(int *output){
    memcpy(output,SpkLiq_pos_i,sizeof(int)*SpkLiq_inh_connections);
}

// NEW REALLOC
void writes_pre_i_weights(float *output, int number_of_connections){
  SpkLiq_w_i = realloc(SpkLiq_w_i,sizeof(float)*number_of_connections);
  if (SpkLiq_w_i==NULL)
  {
    fprintf(stderr,"SpkLiq_w_i - realloc error\n");
    exit(EXIT_FAILURE);
  }
  memcpy(SpkLiq_w_i,output,sizeof(float)*number_of_connections); // overwrites the inhibitory PRE connections
}

void reads_pre_i_weights(float *output){
    memcpy(output,SpkLiq_w_i,sizeof(float)*SpkLiq_inh_connections);
}

// NEW REALLOC
void writes_pre_e_connections(int *output, int number_of_connections){
  SpkLiq_pre_e = realloc(SpkLiq_pre_e,sizeof(int)*number_of_connections);
  if (SpkLiq_pre_e==NULL)
  {
    fprintf(stderr,"SpkLiq_pre_e - realloc error\n");
    exit(EXIT_FAILURE);
  }
  memcpy(SpkLiq_pre_e,output,sizeof(int)*number_of_connections); // overwrites the inhibitory PRE connections
}

void reads_pre_e_connections(int *output){
    memcpy(output,SpkLiq_pre_e,sizeof(int)*SpkLiq_exc_connections);
}

// NEW REALLOC
void writes_pos_e_connections(int *output, int number_of_connections){
  SpkLiq_pos_e = realloc(SpkLiq_pos_e,sizeof(int)*number_of_connections);
  if (SpkLiq_pos_e==NULL)
  {
    fprintf(stderr,"SpkLiq_pos_e - realloc error\n");
    exit(EXIT_FAILURE);
  }
  memcpy(SpkLiq_pos_e,output,sizeof(int)*number_of_connections); // overwrites the inhibitory PRE connections
}

void reads_pos_e_connections(int *output){
    memcpy(output,SpkLiq_pos_e,sizeof(int)*SpkLiq_exc_connections);
}

// NEW REALLOC
void writes_pre_e_weights(float *output, int number_of_connections){
  SpkLiq_w_e = realloc(SpkLiq_w_e,sizeof(float)*number_of_connections);
  if (SpkLiq_w_e==NULL)
  {
    fprintf(stderr,"SpkLiq_w_e - realloc error\n");
    exit(EXIT_FAILURE);
  }
  memcpy(SpkLiq_w_e,output,sizeof(float)*number_of_connections); // overwrites the inhibitory PRE connections
}

void reads_pre_e_weights(float *output){
    memcpy(output,SpkLiq_w_e,sizeof(float)*SpkLiq_exc_connections);
}

void writes_inhibitory_indices(int *output){
    memcpy(SpkLiq_inhibitory_indices,output,sizeof(int)*SpkLiq_number_of_inh_neurons); // overwrites the inhibitory indices
}

void reads_inhibitory_indices(int *output){
    memcpy(output,SpkLiq_inhibitory_indices,sizeof(int)*SpkLiq_number_of_inh_neurons);
}

void writes_excitatory_indices(int *output){
    memcpy(SpkLiq_excitatory_indices,output,sizeof(int)*SpkLiq_number_of_exc_neurons); // overwrites the excitatory indices
}

void reads_excitatory_indices(int *output){
    memcpy(output,SpkLiq_excitatory_indices,sizeof(int)*SpkLiq_number_of_exc_neurons);
}

void writes_refrac_values(float *output){
    memcpy(SpkLiq_refrac_values,output,sizeof(float)*SpkLiq_number_of_neurons);
}

void reads_refrac_values(float *output){
    memcpy(output,SpkLiq_refrac_values,sizeof(float)*SpkLiq_number_of_neurons);
}

void writes_noisy_offset_currents(float *output){
    memcpy(SpkLiq_noisy_offset_currents,output,sizeof(float)*SpkLiq_number_of_neurons);
}

void reads_noisy_offset_currents(float *output){
    memcpy(output,SpkLiq_noisy_offset_currents,sizeof(float)*SpkLiq_number_of_neurons);
}

void reads_noisy_currents(float *output){
    memcpy(output,SpkLiq_noisy_currents,sizeof(float)*SpkLiq_number_of_neurons);
}

void writes_connected(int *output){
    memcpy(SpkLiq_neurons_connected,output,sizeof(int)*SpkLiq_number_of_neurons);
    // indicates with 1 if the neuron has a connection to other neuron
}

void reads_connected(int *output){
    memcpy(output,SpkLiq_neurons_connected,sizeof(int)*SpkLiq_number_of_neurons);
    // indicates with 1 if the neuron has a connection to other neuron
}

void writes_SpkLiq_inh_connections(const int input_value){
    SpkLiq_inh_connections=input_value;
}

void writes_SpkLiq_exc_connections(const int input_value){
    SpkLiq_exc_connections=input_value;
}


void change_liquid_parameters(float *output){
    int c = 0;
    for(int i=0; i<2; i++)
        for(int j=0; j<2; j++)
            for(int k=0; k<6; k++)
            {
                SpkLiq_liquid_parameters[i][j][k]=output[c];
                c++;
            }

}

void stats_liquid(int *output){
    output[0]=SpkLiq_number_of_inh_neurons;
    output[1]=SpkLiq_number_of_exc_neurons;
    output[2]=SpkLiq_inh_connections;
    output[3]=SpkLiq_exc_connections;
}

void external_update(const int *restrict const exc_inputs,
                     const int *restrict const inh_inputs,
                     const float *restrict const exc_weights,
                     const float *restrict const inh_weights,
                     const int *restrict const size_exc,
                     const int *restrict const size_inh,
                     int *restrict const output_spikes,
                     const int total_runs)
{
  /*
  *exc_inputs:  pointer to the array with the excitatory spikes (to save memory, it's a concatenated array with all the values)
  *exc_weights: pointer to the array with the excitatory weights
  *inh_inputs:  pointer to the array with the inhibitory spikes
  *inh_weights: pointer to the array with the inhibitory weights
  *size_exc: pointer to the array with the sizes of the excitatory ones (offsets to access the spikes and weights)
  *size_inh: pointer to the array with the sizes of the inhibitory ones
  *output_spikes: pointer to the arrays where the outputs will be saved (concatenated)
  total_runs: total number of updates do be done

  Note:
  Using ctypes, numpy arrays always look like a vector in C, not a matrix (even if the numpy array IS a matrix)
  The reason is that numpy matrix are always arrays also inside numpy :)

  */

  int offset_exc=0;
  int offset_inh=0;

  for(int i=0; i<(total_runs); i++){

    SpikingLiquid_update(exc_inputs+sizeof(int)*offset_exc,     inh_inputs+sizeof(int)*offset_inh,
                         exc_weights+sizeof(float)*offset_exc,  inh_weights+sizeof(float)*offset_inh,
                         size_exc[i], size_inh[i]);
    offset_exc += size_exc[i];
    offset_inh += size_inh[i];

    memcpy(output_spikes+SpkLiq_number_of_neurons*i,SpkLiq_test_vthres,sizeof(int)*SpkLiq_number_of_neurons); // saves the spikes output from the liquid

  }
}

void free_all(){
/*
    Allocated memory (I need to check if I could free some positions earlier)
*/
    if (SpkLiq_initialized) //Checks if the simulator was already initialized
    {
      // init allocations
      if (SpkLiq_number_of_neurons_slice)
        free(SpkLiq_number_of_neurons_slice); //stores the number of neurons each thread are going to process

      if (SpkLiq_neurons_membrane)
        free(SpkLiq_neurons_membrane); //stores the membrane potentials for all neurons

      if (SpkLiq_neurons_membrane_init)
        free(SpkLiq_neurons_membrane_init); //stores the INITIAL membrane potentials for all neurons

      if (SpkLiq_neurons_exc_curr)
        free(SpkLiq_neurons_exc_curr); //excitatory currents levels for all neurons

      if (SpkLiq_neurons_inh_curr)
        free(SpkLiq_neurons_inh_curr); //inhibitory currents levels for all neurons

      if (SpkLiq_refrac_timer)
        free(SpkLiq_refrac_timer); //timers used to control the refractory period

      if (SpkLiq_test_vthres)
        free(SpkLiq_test_vthres); //controls which neuron is above the voltage threshold and should spike

      if (SpkLiq_noisy_offset_currents)
        free(SpkLiq_noisy_offset_currents); //values of the neuron model constant noisy offset currents

      if (SpkLiq_noisy_currents)
        free(SpkLiq_noisy_currents); //values of the neuron model variable noisy currents

      if (SpkLiq_inhibitory_indices)
        free(SpkLiq_inhibitory_indices); //Indices of the inhibitory neurons inside the liquid

      if (SpkLiq_excitatory_indices)
        free(SpkLiq_excitatory_indices);  //Indices of the excitatory neurons inside the liquid

      if (SpkLiq_liquid_indices_shuffled)
        free(SpkLiq_liquid_indices_shuffled); //used to generate the random SpkLiq_inhibitory_indices/SpkLiq_excitatory_indices

      if (SpkLiq_refrac_values)
        free(SpkLiq_refrac_values); //stores the refractory periods for all neurons in the liquid

      if (SpkLiq_3Didx)
        free(SpkLiq_3Didx); //stores the liquid's 3D structure according to the neuron index

      if (SpkLiq_spike_time)
        free(SpkLiq_spike_time);  //stores the last time each neuron spiked

      if (SpkLiq_neurons_connected)
        free(SpkLiq_neurons_connected); //indicates if the neuron has a connection

      if (SpkLiq_threads_ptrs)
        free(SpkLiq_threads_ptrs); //stores the pointers used with the pthread_create

      if (SpkLiq_thread_id)
        free(SpkLiq_thread_id); //stores the thread ids used with the pthread_create

      if (SpkLiq_number_of_inh_connections_thread)
        free(SpkLiq_number_of_inh_connections_thread); //stores how many inhibitory connections each thread will process

      if (SpkLiq_number_of_exc_connections_thread)
        free(SpkLiq_number_of_exc_connections_thread); //stores how many excitatory connections each thread will process



      /* These variables are freed inside their parent functions
      distance_results

      //but this ones are initialized as arrays of pointers inside init
      SpkLiq_pre_i_thread
      SpkLiq_pos_i_thread
      SpkLiq_w_i_thread
      SpkLiq_pre_e_thread
      SpkLiq_pos_e_thread
      SpkLiq_w_e_thread
      //end

          SpkLiq_pre_i_thread_internal //This variable points to the same memory position as SpkLiq_pre_i_thread
          SpkLiq_pos_i_thread_internal //This variable points to the same memory position as SpkLiq_pos_i_thread
          SpkLiq_w_i_thread_internal //This variable points to the same memory position as SpkLiq_w_i_thread

          SpkLiq_pre_e_thread_internal //This variable points to the same memory position as SpkLiq_pre_e_thread
          SpkLiq_pos_e_thread_internal //This variable points to the same memory position as SpkLiq_pos_e_thread
          SpkLiq_w_e_thread_internal //This variable points to the same memory position as SpkLiq_w_e_thread

      //but this ones are initialized as arrays of pointers inside init
      SpkLiq_inh_connections_thread
      SpkLiq_exc_connections_thread
      //end

      free(SpkLiq_spiked_i); //stores indices of the inhibitory neurons that spiked in the last simulation SpkLiq_step
      free(SpkLiq_spiked_e); //stores indices of the excitatory neurons that spiked in the last simulation SpkLiq_step

      */


      if (SpkLiq_connected) //verifies if the connections where processed
      {
          free(SpkLiq_number_of_inh_neurons_thread); //stores the number of inhibitory neurons each thread is going to receive
          free(SpkLiq_number_of_inh_neurons_thread_total); //stores the number of inhibitory neurons each thread is going to receive
          free(SpkLiq_number_of_exc_neurons_thread); //stores the number of excitatory neurons each thread is going to receive
          free(SpkLiq_number_of_exc_neurons_thread_total); //stores the number of excitatory neurons each thread is going to receive
          free(SpkLiq_receive_spike_i_idx_thread); //stores how many items are inside each of the above arrays (each thread is going to generate an variable size)
          free(SpkLiq_receive_spike_e_idx_thread); //stores how many items are inside each of the above arrays (each thread is going to generate an variable size)

          //this one is initialized as arrays of pointers inside init
          for(int i=0; i<5; i++)
            free(SpkLiq_threads_states[i]);


          //this ones are initialized as arrays of pointers inside init
          for (int i=0; i<SpkLiq_threads_N; i++){
            // Cleans the variables used by the individual threads
            free(SpkLiq_receive_spike_i_thread[i]);
            free(SpkLiq_receive_spike_i_w_thread[i]);
            free(SpkLiq_receive_spike_e_thread[i]);
            free(SpkLiq_receive_spike_e_w_thread[i]);
            free(SpkLiq_neurons_inh_injections[i]);
            free(SpkLiq_neurons_exc_injections[i]);
            free(SpkLiq_neurons_exc_injections_w[i]);
            free(SpkLiq_neurons_inh_injections_w[i]);
          }

          // Cleans the variables used to keep the pointers above
          free(SpkLiq_receive_spike_i_thread); //stores the indices of which liquid's inhibitory neuron receives a spike (output from the threads)
          free(SpkLiq_receive_spike_i_w_thread); //stores the weights of the above received spikes
          free(SpkLiq_receive_spike_e_thread); //stores the indices of which liquid's excitatory neuron receives a spike (output from the threads)
          free(SpkLiq_receive_spike_e_w_thread); //stores the weights of the above received spikes

          free(SpkLiq_pre_i); //stores the indices of the inhibitory=>? pre-synaptic connections in the liquid
          free(SpkLiq_pos_i); //stores the indices of the inhibitory=>? pos-synaptic connections in the liquid
          free(SpkLiq_w_i); //stores the weights of the connections above
          free(SpkLiq_pre_e); //stores the indices of the inhibitory=>? pre-synaptic connections in the liquid
          free(SpkLiq_pos_e); //stores the indices of the inhibitory=>? pos-synaptic connections in the liquid
          free(SpkLiq_w_e); //stores the weights of the connections above

        }

        if (SpkLiq_neurons_inh_injections)
          free(SpkLiq_neurons_inh_injections);

        if (SpkLiq_neurons_exc_injections)
          free(SpkLiq_neurons_exc_injections);

        if (SpkLiq_neurons_exc_injections_w)
          free(SpkLiq_neurons_exc_injections_w);

        if (SpkLiq_neurons_inh_injections_w)
          free(SpkLiq_neurons_inh_injections_w);

        if (SpkLiq_neurons_exc_injections_total)
          free(SpkLiq_neurons_exc_injections_total); //stores the cummulative total number of connections according to the neurons index

        if (SpkLiq_neurons_inh_injections_total)
          free(SpkLiq_neurons_inh_injections_total); //stores the cummulative total number of connections according to the neurons index

        SpkLiq_initialized = 0; //tells Python the main variables are not available
        SpkLiq_connected = 0;  //tells Python the connection variables are not available
  }

}


int check_init()
{
  // Tells Python the state of the simulator's initialization
  return SpkLiq_initialized;
}

int check_connected()
{
  // Tells Python the state of the simulator's connections
  return SpkLiq_connected;
}


int parse_sim_config(char *file_name)
{
    FILE *config_file;

    config_file = fopen(file_name, "r");
    if (config_file==NULL)
    {
        fprintf(stderr, "error: openning config file!\n");
        return EXIT_FAILURE;
    }

    char str_read[80];

    int seeds_not_supplied=1;

    printf("Processing the config file: %s\n", file_name);

    while (fscanf(config_file, "%s\n", str_read) != EOF)
    {
        if (strcmp("#SpkLiq_net_shape",str_read)==0)
        {
              // int SpkLiq_net_shape[3];
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_net_shape[0] = atoi(str_read);
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_net_shape[1] = atoi(str_read);
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_net_shape[2] = atoi(str_read);
              printf("#SpkLiq_net_shape:%d,%d,%d\n",SpkLiq_net_shape[0],SpkLiq_net_shape[1],SpkLiq_net_shape[2]);
              continue;
        }

        if (strcmp("#SpkLiq_lbd_value",str_read)==0)
        {
              // float SpkLiq_lbd_value;
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_lbd_value = atof(str_read);
              printf("#SpkLiq_lbd_value:%f\n",SpkLiq_lbd_value);
              continue;
        }

        if (strcmp("#SpkLiq_step",str_read)==0)
        {
              // float SpkLiq_step;
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_step = atof(str_read);
              printf("#SpkLiq_step:%3.2E\n",SpkLiq_step);
              continue;
        }

        if (strcmp("#SpkLiq_taum",str_read)==0)
        {
              // float SpkLiq_taum;
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_taum = atof(str_read);
              printf("#SpkLiq_taum:%3.2E\n",SpkLiq_taum);
              continue;
        }

        if (strcmp("#SpkLiq_cm",str_read)==0)
        {
              // float SpkLiq_cm;
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_cm = atof(str_read);
              printf("#SpkLiq_cm:%3.2E\n",SpkLiq_cm);
              continue;
        }

        if (strcmp("#SpkLiq_taue",str_read)==0)
        {
              // float SpkLiq_taue;
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_taue = atof(str_read);
              printf("#SpkLiq_taue:%3.2E\n",SpkLiq_taue);
              continue;
        }

        if (strcmp("#SpkLiq_taui",str_read)==0)
        {
              // float SpkLiq_taui;
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_taui = atof(str_read);
              printf("#SpkLiq_taui:%3.2E\n",SpkLiq_taui);
              continue;
        }

        if (strcmp("#SpkLiq_membrane_rand",str_read)==0)
        {
              // float SpkLiq_membrane_rand[2];
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_membrane_rand[0] = atof(str_read);
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_membrane_rand[1] = atof(str_read);
              printf("#SpkLiq_membrane_rand:%3.2E,%3.2E\n",SpkLiq_membrane_rand[0],SpkLiq_membrane_rand[1]);
              continue;
        }

        if (strcmp("#SpkLiq_current_rand",str_read)==0)
        {
              // float SpkLiq_current_rand[2];
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_current_rand[0] = atof(str_read);
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_current_rand[1] = atof(str_read);
              printf("#SpkLiq_current_rand:%3.2E,%3.2E\n",SpkLiq_current_rand[0],SpkLiq_current_rand[1]);
              continue;
        }

        if (strcmp("#SpkLiq_noisy_current_rand",str_read)==0)
        {
              // float SpkLiq_noisy_current_rand;
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_noisy_current_rand = atof(str_read);
              printf("#SpkLiq_noisy_current_rand:%3.2E\n",SpkLiq_noisy_current_rand);
              continue;
        }

        if (strcmp("#SpkLiq_vresets",str_read)==0)
        {
              // float SpkLiq_vresets[2];
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_vresets[0] = atof(str_read);
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_vresets[1] = atof(str_read);
              printf("#SpkLiq_vresets:%3.2E,%3.2E\n",SpkLiq_vresets[0],SpkLiq_vresets[1]);
              continue;
        }

        if (strcmp("#SpkLiq_vthres",str_read)==0)
        {
              // float SpkLiq_vthres;
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_vthres = atof(str_read);
              printf("#SpkLiq_vthres:%3.2E\n",SpkLiq_vthres);
              continue;
        }

        if (strcmp("#SpkLiq_vrest",str_read)==0)
        {
              // float SpkLiq_vrest;
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_vrest = atof(str_read);
              printf("#SpkLiq_vrest:%3.2E\n",SpkLiq_vrest);
              continue;
        }

        if (strcmp("#SpkLiq_inhibitory_percentage",str_read)==0)
        {
              // float SpkLiq_inhibitory_percentage;
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_inhibitory_percentage = atof(str_read);
              printf("#SpkLiq_inhibitory_percentage:%3.2E\n",SpkLiq_inhibitory_percentage);
              continue;
        }

        if (strcmp("#SpkLiq_refractory",str_read)==0)
        {
              // float SpkLiq_refractory[2];
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_refractory[0] = atof(str_read);
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_refractory[1] = atof(str_read);
              printf("#SpkLiq_refractory:%3.2E,%3.2E\n",SpkLiq_refractory[0],SpkLiq_refractory[1]);
              continue;
        }

        if (strcmp("#SpkLiq_user_seeds",str_read)==0)
        {
              // unsigned int SpkLiq_user_seeds[5];
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_user_seeds[0] = atoi(str_read);
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_user_seeds[1] = atoi(str_read);
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_user_seeds[2] = atoi(str_read);
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_user_seeds[3] = atoi(str_read);
              fscanf(config_file, "%s\n", str_read);
              SpkLiq_user_seeds[4] = atoi(str_read);
              printf("#SpkLiq_user_seeds:%u,%u,%u,%u,%u\n",SpkLiq_user_seeds[0],SpkLiq_user_seeds[1],SpkLiq_user_seeds[2],SpkLiq_user_seeds[3],SpkLiq_user_seeds[4]);
              seeds_not_supplied=0;
              continue;
        }

    }

    fclose(config_file);


    // If the user is not supplying the seeds, the simulator generates themn automatically.
    if(seeds_not_supplied)
    {
      rk_state internal_states;
      rk_randomseed(&internal_states); //initializes the random state
      for(int i=0; i<5; i++)
      {
        SpkLiq_user_seeds[i] = rk_interval(9999999999, &internal_states);
      }
      printf("#SpkLiq_user_seeds(random):%u,%u,%u,%u,%u\n",SpkLiq_user_seeds[0],SpkLiq_user_seeds[1],SpkLiq_user_seeds[2],SpkLiq_user_seeds[3],SpkLiq_user_seeds[4]);
    }

    return EXIT_SUCCESS;
}



int save_load_connections_file(char *file_name)
{
    //
    // READS / SAVES the connection structure
    //
    double time_begin, time_end;

    FILE *connections_file;


    connections_file = fopen(file_name, "rb");

    if (connections_file==NULL)
    {
        printf("Calling SpikingLiquid_generate_connections...\n");
        printf("Using the mininum distance of: %f (%2.2f%% probability)\n",sqrt(-log(SpkLiq_min_perc)*SpkLiq_lbd_value*SpkLiq_lbd_value),SpkLiq_min_perc*100);
        time_begin=clock_my_gettime();
        SpikingLiquid_generate_connections();
        time_end=clock_my_gettime();
        printf("SpikingLiquid_generate_connections total time (ms):%f\n", (time_end-time_begin)/1E6);

        connections_file = fopen(file_name, "wb");
        if (!connections_file)
        {
          fprintf(stderr, "error: creating connections file!\n");
          exit(EXIT_FAILURE);
        }

        printf("Saving connections to %s...",file_name);

        fwrite(&SpkLiq_number_of_inh_neurons, sizeof(int), 1, connections_file);
        // - int *SpkLiq_inhibitory_indices: array with the neuron's indices belonging to inhibitory group
        fwrite(SpkLiq_inhibitory_indices, sizeof(int), SpkLiq_number_of_inh_neurons, connections_file);

        fwrite(&SpkLiq_number_of_exc_neurons, sizeof(int), 1, connections_file);
        // - int *SpkLiq_excitatory_indices: array with the neuron's indices belonging to excitatory group
        fwrite(SpkLiq_excitatory_indices, sizeof(int), SpkLiq_number_of_exc_neurons, connections_file);

        // - int SpkLiq_inh_connections: total number of inhibitory=>? connections
        fwrite(&SpkLiq_inh_connections, sizeof(int), 1, connections_file);
        // - int *SpkLiq_pre_i: stores the indices of the inhibitory=>? PRE-synaptic connections in the liquid
        // - int *SpkLiq_pos_i: stores the indices of the inhibitory=>? POS-synaptic connections in the liquid
        // - float *SpkLiq_w_i: stores the weights of the inhibitory connections above
        fwrite(SpkLiq_pre_i, sizeof(int), SpkLiq_inh_connections, connections_file);
        fwrite(SpkLiq_pos_i, sizeof(int), SpkLiq_inh_connections, connections_file);
        fwrite(SpkLiq_w_i, sizeof(float), SpkLiq_inh_connections, connections_file);

        // - int SpkLiq_exc_connections: total number of excitatory=>? connections
        fwrite(&SpkLiq_exc_connections, sizeof(int), 1, connections_file);
        // - int *SpkLiq_pre_e: stores the indices of the excitatory=>? PRE-synaptic connections in the liquid
        // - int *SpkLiq_pos_e: stores the indices of the excitatory=>? POS-synaptic connections in the liquid
        // - float *SpkLiq_w_e: stores the weights of the excitatory connections above
        fwrite(SpkLiq_pre_e, sizeof(int), SpkLiq_exc_connections, connections_file);
        fwrite(SpkLiq_pos_e, sizeof(int), SpkLiq_exc_connections, connections_file);
        fwrite(SpkLiq_w_e, sizeof(float), SpkLiq_exc_connections, connections_file);

        fwrite(&SpkLiq_number_of_neurons, sizeof(int), 1, connections_file);
        // - int *restrict SpkLiq_neurons_connected; //Signalizes that this neuron has at least ONE connection to other neuron
        fwrite(SpkLiq_neurons_connected, sizeof(int), SpkLiq_number_of_neurons, connections_file);

        fclose (connections_file);
        printf("DONE!\n");

    }else
    {
        printf("Loading connections from %s...",file_name);
        time_begin=clock_my_gettime();

        fread(&SpkLiq_number_of_inh_neurons, sizeof(int), 1, connections_file);
        // - int *SpkLiq_inhibitory_indices: array with the neuron's indices belonging to inhibitory group
        fread(SpkLiq_inhibitory_indices, sizeof(int), SpkLiq_number_of_inh_neurons, connections_file);

        fread(&SpkLiq_number_of_exc_neurons, sizeof(int), 1, connections_file);
        // - int *SpkLiq_excitatory_indices: array with the neuron's indices belonging to excitatory group
        fread(SpkLiq_excitatory_indices, sizeof(int), SpkLiq_number_of_exc_neurons, connections_file);

        // - int SpkLiq_inh_connections: total number of inhibitory=>? connections
        fread(&SpkLiq_inh_connections, sizeof(int), 1, connections_file);

        // Allocates the memory according to the totals generated above
        SpkLiq_pre_i = malloc(sizeof(int)*SpkLiq_inh_connections);
        SpkLiq_pos_i = malloc(sizeof(int)*SpkLiq_inh_connections);
        SpkLiq_w_i = malloc(sizeof(float)*SpkLiq_inh_connections);
        if (SpkLiq_pre_i==NULL || SpkLiq_pos_i==NULL || SpkLiq_w_i==NULL)
        {
            fprintf(stderr,"Loading connections INH -  malloc ERROR!");
            exit(EXIT_FAILURE); //I need to check which error I should signal here...
        }

        // - int *SpkLiq_pre_i: stores the indices of the inhibitory=>? PRE-synaptic connections in the liquid
        // - int *SpkLiq_pos_i: stores the indices of the inhibitory=>? POS-synaptic connections in the liquid
        // - float *SpkLiq_w_i: stores the weights of the inhibitory connections above
        fread(SpkLiq_pre_i, sizeof(int), SpkLiq_inh_connections, connections_file);
        fread(SpkLiq_pos_i, sizeof(int), SpkLiq_inh_connections, connections_file);
        fread(SpkLiq_w_i, sizeof(float), SpkLiq_inh_connections, connections_file);

        // - int SpkLiq_exc_connections: total number of excitatory=>? connections
        fread(&SpkLiq_exc_connections, sizeof(int), 1, connections_file);

        // Allocates the memory according to the totals generated above
        SpkLiq_pre_e = malloc(sizeof(int)*SpkLiq_exc_connections);
        SpkLiq_pos_e = malloc(sizeof(int)*SpkLiq_exc_connections);
        SpkLiq_w_e = malloc(sizeof(float)*SpkLiq_exc_connections);
        if (SpkLiq_pre_e==NULL || SpkLiq_pos_e==NULL || SpkLiq_w_e==NULL)
        {
            fprintf(stderr,"Loading connections EXC -  malloc ERROR!");
            exit(EXIT_FAILURE); //I need to check which error I should signal here...
        }

        // - int *SpkLiq_pre_e: stores the indices of the excitatory=>? PRE-synaptic connections in the liquid
        // - int *SpkLiq_pos_e: stores the indices of the excitatory=>? POS-synaptic connections in the liquid
        // - float *SpkLiq_w_e: stores the weights of the excitatory connections above
        fread(SpkLiq_pre_e, sizeof(int), SpkLiq_exc_connections, connections_file);
        fread(SpkLiq_pos_e, sizeof(int), SpkLiq_exc_connections, connections_file);
        fread(SpkLiq_w_e, sizeof(float), SpkLiq_exc_connections, connections_file);

        fread(&SpkLiq_number_of_neurons, sizeof(int), 1, connections_file);
        // - int *restrict SpkLiq_neurons_connected; //Signalizes that this neuron has at least ONE connection to other neuron
        fread(SpkLiq_neurons_connected, sizeof(int), SpkLiq_number_of_neurons, connections_file);

        fclose (connections_file);

        printf("DONE!\n");

        time_end=clock_my_gettime();
        printf("Loading connections total time (ms):%f\n", (time_end-time_begin)/1E6);

     }

  return 0; //Indicates that everything went just fine :D
}


int generate_input_list_from_bits(const long int *restrict const input_bytes, int *restrict output, const int length)
{
    int output_idx=0;
    for(int i=0; i<length; i++){
        for(int j=0; j<64; j++)
        {
            if (input_bytes[i] & (1L<<(j&(64-1))))
            {
                output[output_idx]=i*64+j;
                // printf("output[%d]=%d\n",output_idx,output[output_idx]);
                output_idx++;
            }
        }
    }
    return output_idx;
}



//
// The simulator depends on the input arguments passed by the user to run properly.
//
int main(int argc, char *argv[]){

    double time_begin, time_end;

    char file_suffix_config[] = "_sim_config.txt";
    char file_suffix_connections[] = "_sim_connections.bin";
    char file_suffix_exc_inputs[] = "_sim_exc_inputs.bin";
    char file_suffix_exc_inputs_weights[] = "_sim_exc_inputs_weights.bin";
    char file_suffix_states[] = "_sim_states.bin";
    char file_suffix_rkstates[] = "_sim_rkstates.bin";
    char file_suffix_outputs[] = "_sim_outputs.bin";

    char *file_name;
    char *file_base_name;

    int number_of_iterations = 0; //This is an argument passed by the user

    int input_fd;
    long int *input_data;
    struct stat input_sbuf;
    int *input_spikes;

    int output_fd;

    int input_weights_fd;
    float *input_spikes_exc_w;

    int states_fd;

    int rkstates_fd;

    int number_of_spikes_input = 0;
    unsigned long int offset = 0;


    if (argc < 4)
    {
        fprintf(stderr,"Usage: %s N file_name_base M -i -s -r -c -p\nWhere:\n N is no. of thread\n file_name_base is the string used to name the files\n M is the number of iterations\n -i reads the inputs from a file (otherwise it will run with no input spikes).\n -s indicates the last states should be loaded from file.\n -r indicates the last random states should be loaded from file (you must use the same number of threads in this case!).\n -c indicates not to print the number of spikes each step produced.\n -p indicates the main is called inside Python.\n -o generates the output file.\n", argv[0]);
        // exit(EXIT_FAILURE);
        return 1;
    }


    // Verifies the options
    int read_states = 0;
    int read_rk_states = 0;
    int print_spikes = 1;
    int inputs_switch = 0;
    int python_switch = 0;
    int output_switch = 0;
    for(int i=4; i<argc; i++)
    {
      char *arg2str=argv[i];
      char s_arg[]="-s";
      char r_arg[]="-r";
      char c_arg[]="-c";
      char p_arg[]="-p";
      char i_arg[]="-i";
      char o_arg[]="-o";

      if(strcmp(arg2str,s_arg)==0)
          read_states = 1;

      if(strcmp(arg2str,r_arg)==0)
          read_rk_states = 1;

      if(strcmp(arg2str,c_arg)==0)
          print_spikes = 0;

      if(strcmp(arg2str,p_arg)==0)
          python_switch = 1;

      if(strcmp(arg2str,i_arg)==0)
          inputs_switch = 1;

      if(strcmp(arg2str,o_arg)==0)
          output_switch = 1;
    }

    SpkLiq_threads_N = atoi(argv[1]); //user input total number of threads

    if ((SpkLiq_threads_N < 2) || (SpkLiq_threads_N > MAX_THREAD)) // loop_start has this problem: [SpkLiq_tid-1]
    {
        fprintf(stderr,"The no of thread should between 2 and %d.\n", MAX_THREAD);
        if(python_switch)
        {
          return 1;
        }
        else
        {
          exit(EXIT_FAILURE);
        }
    }

    // Sets the pointer to the user command line input
    file_base_name = argv[2];

    //
    // READS/PARSES the configuration file
    //
    file_name = malloc(strlen(file_base_name)+strlen(file_suffix_config)+1);//+1 for the zero-terminator
    strcpy(file_name, file_base_name);
    strcat(file_name, file_suffix_config);

    printf("Initializing the input variables...\n");
    if (parse_sim_config(file_name))
//    if (parse_sim_config("rdc2_experiment1/rdc2_sim_config.txt"))
    {
        fprintf(stderr,"Error whilst parsing the %s_sim_config.txt file!\n", file_base_name);
        if(python_switch)
        {
          return 1;
        }
        else
        {
          exit(EXIT_FAILURE);
        }
    }
    free(file_name);


    printf("Calling SpikingLiquid_init...\n");
    time_begin=clock_my_gettime();

    SpikingLiquid_init();


    time_end=clock_my_gettime();
    printf("SpikingLiquid_init total time (ms):%f\n", (time_end-time_begin)/1E6);


    //
    // Generates and save OR simply load the connections from/to a file
    //
    file_name = malloc(strlen(file_base_name)+strlen(file_suffix_connections)+1);//+1 for the zero-terminator
    strcpy(file_name, file_base_name);
    strcat(file_name, file_suffix_connections);
    printf("Initializing the connections...\n");
    if (save_load_connections_file(file_name))
//    if (save_load_connections_file("rdc2_experiment1/rdc2_sim_connections.bin"))
    {
        fprintf(stderr,"Error whilst saving/loading %s_sim_connections.bin file!\n", file_base_name);
        if(python_switch)
        {
          return 1;
        }
        else
        {
          exit(EXIT_FAILURE);
        }
    }
    free(file_name);

    //
    // Process the connections
    //
    printf("Calling SpikingLiquid_process_connections...\n");
    time_begin=clock_my_gettime();

    SpikingLiquid_process_connections();

    time_end=clock_my_gettime();
    printf("SpikingLiquid_process_connections total time (ms):%f\n", (time_end-time_begin)/1E6);



    // Sets the number of iterations (make sure the input file has the same number of input arrays)
    number_of_iterations = atoi(argv[3]);




    //
    // CREATES THE OUTPUT FILE
    //
    if(output_switch)
    {
        file_name = malloc(strlen(file_base_name)+strlen(file_suffix_outputs)+1);//+1 for the zero-terminator
        strcpy(file_name, file_base_name);
        strcat(file_name, file_suffix_outputs);

        if ((output_fd = open(file_name, O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)) == -1) {
            fprintf(stderr, "error: creating %s output file!\n Probably you need to erase the old file...\n", file_name);
            if(python_switch)
            {
              return 1;
            }
            else
            {
              exit(EXIT_FAILURE);
            }
        }
        free(file_name);
    }


    if(inputs_switch)
    {
      //
      // OPENS THE INPUT FILE
      //
      file_name = malloc(strlen(file_base_name)+strlen(file_suffix_exc_inputs)+1);//+1 for the zero-terminator
      strcpy(file_name, file_base_name);
      strcat(file_name, file_suffix_exc_inputs);

      input_spikes = malloc(SpkLiq_number_of_neurons*sizeof(int));

      if ((input_fd = open(file_name, O_RDONLY)) == -1) {
          fprintf(stderr, "error: openning %s exc input file!\n", file_name);
          if(python_switch)
          {
            return 1;
          }
          else
          {
            exit(EXIT_FAILURE);
          }
      }

      if (stat(file_name, &input_sbuf) == -1) {
          fprintf(stderr, "error: stat %s exc input file!\n", file_name);
          if(python_switch)
          {
            return 1;
          }
          else
          {
            exit(EXIT_FAILURE);
          }
      }

      input_data = (long int *)mmap(NULL, input_sbuf.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, input_fd, 0);

      if (input_data == MAP_FAILED) {
          fprintf(stderr, "error: input mmap!\n");
          if(python_switch)
          {
            return 1;
          }
          else
          {
            exit(EXIT_FAILURE);
          }
      }
      free(file_name);



      //
      // READS THE INPUT WEIGHTS FILE
      //
      file_name = malloc(strlen(file_base_name)+strlen(file_suffix_exc_inputs_weights)+1);//+1 for the zero-terminator
      strcpy(file_name, file_base_name);
      strcat(file_name, file_suffix_exc_inputs_weights);

      input_spikes_exc_w = malloc(SpkLiq_number_of_neurons*sizeof(float));

      if ((input_weights_fd = open(file_name, O_RDONLY)) == -1) {
//      if ((input_weights_fd = open("rdc2_experiment1/experiment_1_weights.bin", O_RDONLY)) == -1) {
          fprintf(stderr, "error: openning %s exc input weights file!\n", file_name);
          if(python_switch)
          {
            return 1;
          }
          else
          {
            exit(EXIT_FAILURE);
          }
      }
      free(file_name);

      read(input_weights_fd, input_spikes_exc_w, sizeof(float)*SpkLiq_number_of_neurons);
      close(input_weights_fd);
    }


    if(read_states){
        //
        // READS THE STATES FILE
        //
        file_name = malloc(strlen(file_base_name)+strlen(file_suffix_states)+1);//+1 for the zero-terminator
        strcpy(file_name, file_base_name);
        strcat(file_name, file_suffix_states);

        if ((states_fd = open(file_name, O_RDONLY)) == -1) {
            fprintf(stderr, "error: openning %s states file!\n Probably you haven't saved this file before...\n", file_name);
            if(python_switch)
            {
              return 1;
            }
            else
            {
              exit(EXIT_FAILURE);
            }
        }
        free(file_name);

        read(states_fd, SpkLiq_neurons_membrane, sizeof(float)*SpkLiq_number_of_neurons);
        // SpkLiq_neurons_membrane = malloc(sizeof(float)*SpkLiq_number_of_neurons);
        read(states_fd, SpkLiq_neurons_membrane_init, sizeof(float)*SpkLiq_number_of_neurons);
        // SpkLiq_neurons_membrane_init = malloc(sizeof(float)*SpkLiq_number_of_neurons);
        read(states_fd, SpkLiq_neurons_exc_curr, sizeof(float)*SpkLiq_number_of_neurons);
        // SpkLiq_neurons_exc_curr = malloc(sizeof(float)*SpkLiq_number_of_neurons);
        read(states_fd, SpkLiq_neurons_inh_curr, sizeof(float)*SpkLiq_number_of_neurons);
        // SpkLiq_neurons_inh_curr = malloc(sizeof(float)*SpkLiq_number_of_neurons);
        read(states_fd, SpkLiq_refrac_timer, sizeof(float)*SpkLiq_number_of_neurons);
        // SpkLiq_refrac_timer = malloc(sizeof(float)*SpkLiq_number_of_neurons);
        read(states_fd, SpkLiq_test_vthres, sizeof(int)*SpkLiq_number_of_neurons);
        // SpkLiq_test_vthres = malloc(sizeof(int)*SpkLiq_number_of_neurons);
        read(states_fd, SpkLiq_test_vthres_bits, sizeof(long int)*SpkLiq_number_of_long_ints);
        // SpkLiq_test_vthres_bits = calloc(SpkLiq_number_of_long_ints,sizeof(long int));
        read(states_fd, SpkLiq_spike_time, sizeof(float)*SpkLiq_number_of_neurons);
        // SpkLiq_spike_time = malloc(sizeof(float)*SpkLiq_number_of_neurons);
        read(states_fd, &SpkLiq_current_time, sizeof(float));
        read(states_fd, &SpkLiq_current_step, sizeof(int));

        close(states_fd);
    }


    if(read_rk_states){
        //
        // READS THE RANDOM STATES FILE
        //
        file_name = malloc(strlen(file_base_name)+strlen(file_suffix_rkstates)+1);//+1 for the zero-terminator
        strcpy(file_name, file_base_name);
        strcat(file_name, file_suffix_rkstates);

        if ((rkstates_fd = open(file_name, O_RDONLY)) == -1) {
            fprintf(stderr, "error: openning %s random states file!\n Probably you haven't saved this file before...\n", file_name);
            if(python_switch)
            {
              return 1;
            }
            else
            {
              exit(EXIT_FAILURE);
            }
        }
        free(file_name);

        for(int i=0; i<5; i++)
          read(rkstates_fd, SpkLiq_threads_states[i], sizeof(rk_state)*SpkLiq_threads_N);

        close(rkstates_fd);
    }


    printf("Current time (ms):%f\n", SpkLiq_current_time);
    printf("Current step:%d\n", SpkLiq_current_step);


    if(inputs_switch)
    {
        if((input_sbuf.st_size/(sizeof(long int)*SpkLiq_number_of_long_ints))<number_of_iterations)
        {
            number_of_iterations = (int) input_sbuf.st_size/(sizeof(long int)*SpkLiq_number_of_long_ints);
        }
    }

    printf("Number_of_iterations:%d\n", number_of_iterations);

    // Gets the initial time in nanoseconds
    time_begin=clock_my_gettime();

    for(int i=0; i<number_of_iterations; i++)
    {
        if(inputs_switch)
        {
          number_of_spikes_input = generate_input_list_from_bits(&(input_data[offset]), input_spikes, SpkLiq_number_of_long_ints);
          SpikingLiquid_update(input_spikes, NULL, input_spikes_exc_w, NULL, number_of_spikes_input, 0);
        }else
        {
          // time_begin=clock_my_gettime();

          SpikingLiquid_update(NULL, NULL, NULL, NULL, 0, 0);

          // time_end=clock_my_gettime();
          // printf("SpikingLiquid_update total time (ms):%f\n", (time_end-time_begin)/1E6);
        }

        if(print_spikes)
        {
            int c=0;
            for(int i=0;i<SpkLiq_number_of_neurons;i++)
            {
                if(SpkLiq_test_vthres[i])
                  c++;
            }
            printf("Number of spikes:%d\n",c);
        }

        if(output_switch)
            write(output_fd, SpkLiq_test_vthres_bits, sizeof(long int)*SpkLiq_number_of_long_ints);

        offset += (long int)SpkLiq_number_of_long_ints;
    }

    // Gets the final time in nanoseconds
    time_end=clock_my_gettime();
    printf("Total update time (ms):%f\n", (time_end-time_begin)/1E6);
    printf("Average update time (ms):%f\n", ((time_end-time_begin)/1E6)/number_of_iterations);


    if(inputs_switch)
    {
      //
      // CLOSES THE INPUT FILE / MMAP
      //
      if (munmap(input_data, input_sbuf.st_size) == -1)
      {
          fprintf(stderr, "error: munmap!\n");
          if(python_switch)
          {
            return 1;
          }
          else
          {
            exit(EXIT_FAILURE);
          }
      }
      close(input_fd);
    }

    //
    // WRITES THE STATES FILE
    //
    file_name = malloc(strlen(file_base_name)+strlen(file_suffix_states)+1);//+1 for the zero-terminator
    strcpy(file_name, file_base_name);
    strcat(file_name, file_suffix_states);

    if ((states_fd = open(file_name, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)) == -1) {
        fprintf(stderr, "error: openning/creating %s states file!\n", file_name);
        if(python_switch)
        {
          return 1;
        }
        else
        {
          exit(EXIT_FAILURE);
        }
    }
    free(file_name);

    write(states_fd, SpkLiq_neurons_membrane, sizeof(float)*SpkLiq_number_of_neurons);
    // SpkLiq_neurons_membrane = malloc(sizeof(float)*SpkLiq_number_of_neurons);
    write(states_fd, SpkLiq_neurons_membrane_init, sizeof(float)*SpkLiq_number_of_neurons);
    // SpkLiq_neurons_membrane_init = malloc(sizeof(float)*SpkLiq_number_of_neurons);
    write(states_fd, SpkLiq_neurons_exc_curr, sizeof(float)*SpkLiq_number_of_neurons);
    // SpkLiq_neurons_exc_curr = malloc(sizeof(float)*SpkLiq_number_of_neurons);
    write(states_fd, SpkLiq_neurons_inh_curr, sizeof(float)*SpkLiq_number_of_neurons);
    // SpkLiq_neurons_inh_curr = malloc(sizeof(float)*SpkLiq_number_of_neurons);
    write(states_fd, SpkLiq_refrac_timer, sizeof(float)*SpkLiq_number_of_neurons);
    // SpkLiq_refrac_timer = malloc(sizeof(float)*SpkLiq_number_of_neurons);
    write(states_fd, SpkLiq_test_vthres, sizeof(int)*SpkLiq_number_of_neurons);
    // SpkLiq_test_vthres = malloc(sizeof(int)*SpkLiq_number_of_neurons);
    write(states_fd, SpkLiq_test_vthres_bits, sizeof(long int)*SpkLiq_number_of_long_ints);
    // SpkLiq_test_vthres_bits = calloc(SpkLiq_number_of_long_ints,sizeof(long int));
    write(states_fd, SpkLiq_spike_time, sizeof(float)*SpkLiq_number_of_neurons);
    // SpkLiq_spike_time = malloc(sizeof(float)*SpkLiq_number_of_neurons);
    write(states_fd, &SpkLiq_current_time, sizeof(float));
    write(states_fd, &SpkLiq_current_step, sizeof(int));

    close(states_fd);

    //
    // WRITES THE RANDOM STATES FILE
    //
    file_name = malloc(strlen(file_base_name)+strlen(file_suffix_rkstates)+1);//+1 for the zero-terminator
    strcpy(file_name, file_base_name);
    strcat(file_name, file_suffix_rkstates);

    if ((rkstates_fd = open(file_name, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)) == -1) {
        fprintf(stderr, "error: openning/creating %s random states file!\n", file_name);
        if(python_switch)
        {
          return 1;
        }
        else
        {
          exit(EXIT_FAILURE);
        }
    }
    free(file_name);

    for(int i=0; i<5; i++)
      write(rkstates_fd, SpkLiq_threads_states[i], sizeof(rk_state)*SpkLiq_threads_N);

    close(rkstates_fd);

    if(output_switch)
        close(output_fd);

    // free_all(); // A good OS doesn't need this because the process is going to finish after this point.

    return 0;
}

// S_IRUSR
// 00400 user has read permission
// S_IWUSR
// 00200 user has write permission
// S_IRGRP
// 00040 group has read permission
// S_IROTH
// 00004 others have read permission


import numpy
import sys
import ctypes
import time
SNNSIM=ctypes.CDLL("libBEE.so")


##
## First the ctypes interfaces to the C shared library:
##

BEE_setup = SNNSIM.user_setup
BEE_setup.restype=ctypes.c_int32
BEE_setup.argtypes = [
                numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS'),
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                numpy.ctypeslib.ndpointer(dtype=numpy.float32, flags='ALIGNED,C_CONTIGUOUS'),
                numpy.ctypeslib.ndpointer(dtype=numpy.float32, flags='ALIGNED,C_CONTIGUOUS'),
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                numpy.ctypeslib.ndpointer(dtype=numpy.float32, flags='ALIGNED,C_CONTIGUOUS'),
                ctypes.c_float,
                ctypes.c_float,
                numpy.ctypeslib.ndpointer(dtype=numpy.uint32, flags='ALIGNED,C_CONTIGUOUS'),
                ctypes.c_int32
               ]

# int user_setup(
#                 int *_SpkLiq_net_shape,
#                 float _SpkLiq_lbd_value,
#                 float _SpkLiq_step,
#                 float _SpkLiq_taum,
#                 float _SpkLiq_cm,
#                 float _SpkLiq_taue,
#                 float _SpkLiq_taui,
#                 float *_SpkLiq_membrane_rand,
#                 float *_SpkLiq_current_rand,
#                 float _SpkLiq_noisy_current_rand,
#                 float _SpkLiq_vresets,
#                 float _SpkLiq_vthres,
#                 float _SpkLiq_vrest,
#                 float *_SpkLiq_refractory,
#                 float _SpkLiq_inhibitory_percentage,
#                 float _SpkLiq_min_perc,
#                 unsigned int *my_seeds,
#                 int _SpkLiq_threads_N
#                 ){


# Makes possible to call the main and then use the same interfaces as the command line version of the simulator
# int main(int argc, char *argv[])
BEE_main = SNNSIM.main
BEE_main.restype=ctypes.c_int32
BEE_main.argtypes = [
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_char_p)
               ]

#
BEE_freeC = SNNSIM.free_all
BEE_freeC.restype = None
BEE_freeC.argtypes =[]

BEE_initialized = SNNSIM.check_init
BEE_initialized.restype = ctypes.c_int32
BEE_initialized.argtypes =[]

BEE_connected = SNNSIM.check_connected
BEE_connected.restype = ctypes.c_int32
BEE_connected.argtypes =[]

liquid_init = SNNSIM.SpikingLiquid_init
liquid_init.restype = None
liquid_init.argtypes =[]

liquid_update = SNNSIM.SpikingLiquid_update
liquid_update.restype = None
liquid_update.argtypes = [
                    numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS'),
                    numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS'),
                    numpy.ctypeslib.ndpointer(dtype=numpy.float32, flags='ALIGNED,C_CONTIGUOUS'),
                    numpy.ctypeslib.ndpointer(dtype=numpy.float32, flags='ALIGNED,C_CONTIGUOUS'),
                    ctypes.c_int32,
                    ctypes.c_int32,
]

liquid_reset = SNNSIM.SpikingLiquid_Reset
liquid_reset.restype = None
liquid_reset.argtypes =[]

liquid_generate_connections = SNNSIM.SpikingLiquid_generate_connections
liquid_generate_connections.restype = None
liquid_generate_connections.argtypes =[]

# NEW
liquid_process_connections = SNNSIM.SpikingLiquid_process_connections
liquid_process_connections.restype = None
liquid_process_connections.argtypes =[]

liquid_output = SNNSIM.reads_test_vthres
liquid_output.restype = None
liquid_output.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS')]

# NEW
liquid_spikes = SNNSIM.reads_spikes
liquid_spikes.restype = ctypes.c_int32
liquid_spikes.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS')]

liquid_voltages = SNNSIM.reads_membranes
liquid_voltages.restype = None
liquid_voltages.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.float32, flags='ALIGNED,C_CONTIGUOUS')]

liquid_initial_voltages = SNNSIM.reads_membranes_init
liquid_initial_voltages.restype = None
liquid_initial_voltages.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.float32, flags='ALIGNED,C_CONTIGUOUS')]

liquid_exc_currents = SNNSIM.reads_exc_synapses
liquid_exc_currents.restype = None
liquid_exc_currents.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.float32, flags='ALIGNED,C_CONTIGUOUS')]

liquid_inh_currents = SNNSIM.reads_inh_synapses
liquid_inh_currents.restype = None
liquid_inh_currents.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.float32, flags='ALIGNED,C_CONTIGUOUS')]

liquid_pre_i = SNNSIM.reads_pre_i_connections
liquid_pre_i.restype = None
liquid_pre_i.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS')]

liquid_pos_i = SNNSIM.reads_pos_i_connections
liquid_pos_i.restype = None
liquid_pos_i.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS')]

liquid_w_i = SNNSIM.reads_pre_i_weights
liquid_w_i.restype = None
liquid_w_i.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.float32, flags='ALIGNED,C_CONTIGUOUS')]

liquid_pre_e = SNNSIM.reads_pre_e_connections
liquid_pre_e.restype = None
liquid_pre_e.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS')]

liquid_pos_e = SNNSIM.reads_pos_e_connections
liquid_pos_e.restype = None
liquid_pos_e.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS')]

liquid_w_e = SNNSIM.reads_pre_e_weights
liquid_w_e.restype = None
liquid_w_e.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.float32, flags='ALIGNED,C_CONTIGUOUS')]

liquid_exc_indices = SNNSIM.reads_excitatory_indices
liquid_exc_indices.restype = None
liquid_exc_indices.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS')]

liquid_inh_indices = SNNSIM.reads_inhibitory_indices
liquid_inh_indices.restype = None
liquid_inh_indices.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS')]

liquid_noisy_offset = SNNSIM.reads_noisy_offset_currents
liquid_noisy_offset.restype = None
liquid_noisy_offset.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.float32, flags='ALIGNED,C_CONTIGUOUS')]

liquid_refrac_values = SNNSIM.reads_refrac_values
liquid_refrac_values.restype = None
liquid_refrac_values.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.float32, flags='ALIGNED,C_CONTIGUOUS')]

liquid_noisy_currents = SNNSIM.reads_noisy_currents
liquid_noisy_currents.restype = None
liquid_noisy_currents.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.float32, flags='ALIGNED,C_CONTIGUOUS')]

liquid_connected = SNNSIM.reads_connected
liquid_connected.restype = None
liquid_connected.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS')]

liquid_stats = SNNSIM.stats_liquid
liquid_stats.restype = None
liquid_stats.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS')]
#     output[0]=SpkLiq_number_of_inh_neurons;
#     output[1]=SpkLiq_number_of_exc_neurons;
#     output[2]=SpkLiq_inh_connections;
#     output[3]=SpkLiq_exc_connections;

# void external_update(const int *restrict const exc_inputs,
#                      const int *restrict const inh_inputs,
#                      const double *restrict const exc_weights,
#                      const double *restrict const inh_weights,
#                      const int *restrict const size_exc,
#                      const int *restrict const size_inh,
#                      int *output_spikes,
#                      const int total_runs)
liquid_ext_update = SNNSIM.external_update
liquid_ext_update.restype = None
liquid_ext_update.argtypes = [
                                numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS'),
                                numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS'),
                                numpy.ctypeslib.ndpointer(dtype=numpy.float32, flags='ALIGNED,C_CONTIGUOUS'),
                                numpy.ctypeslib.ndpointer(dtype=numpy.float32, flags='ALIGNED,C_CONTIGUOUS'),
                                numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS'),
                                numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS'),
                                numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags='ALIGNED,C_CONTIGUOUS'),
                                ctypes.c_int32
                             ]


##
## Python functions to expose the ones ABOVE interfaced using ctypes:
##

def initialize_sim( my_net_shape=[15,3,3], my_lbd_value=1.2, my_seeds=numpy.random.randint(0,10000,5),
                    SpkLiq_step = 0.2E-3,SpkLiq_taum = 30.0E-3,SpkLiq_cm = 30.0E-9,
                    SpkLiq_taue = 3.0E-3,SpkLiq_taui = 6.0E-3,
                    SpkLiq_membrane_rand = [13.5E-3,15.0E-3],
                    SpkLiq_current_rand = [14.975E-9,15.025E-9],
                    SpkLiq_noisy_current_rand = 0.2E-9,
                    SpkLiq_vresets = 13.5E-3,
                    SpkLiq_vthres = 15.0E-3,
                    SpkLiq_vrest = 0.0,
                    SpkLiq_refractory = [3E-3,2E-3],
                    SpkLiq_inhibitory_percentage = 20,
                    SpkLiq_threads_N = 4,
                    SpkLiq_min_perc = 0.01):

    # User setup
    SpkLiq_net_shape = numpy.array(my_net_shape,dtype=numpy.int32)
    SpkLiq_lbd_value = ctypes.c_float(my_lbd_value)


    success = BEE_setup(
                            SpkLiq_net_shape,
                            SpkLiq_lbd_value,
                            ctypes.c_float(SpkLiq_step),
                            ctypes.c_float(SpkLiq_taum),
                            ctypes.c_float(SpkLiq_cm),
                            ctypes.c_float(SpkLiq_taue),
                            ctypes.c_float(SpkLiq_taui),
                            numpy.array(SpkLiq_membrane_rand,dtype=numpy.float32),
                            numpy.array(SpkLiq_current_rand,dtype=numpy.float32),
                            ctypes.c_float(SpkLiq_noisy_current_rand),
                            ctypes.c_float(SpkLiq_vresets),
                            ctypes.c_float(SpkLiq_vthres),
                            ctypes.c_float(SpkLiq_vrest),
                            numpy.array(SpkLiq_refractory,dtype=numpy.float32),
                            ctypes.c_float(SpkLiq_inhibitory_percentage),
                            ctypes.c_float(SpkLiq_min_perc),
                            numpy.array(my_seeds,dtype=numpy.uint32),
                            ctypes.c_int32(SpkLiq_threads_N)
    )

    # Initializes the liquid
    liquid_init()


def simulator_main(cmd_args):
    '''
    Gives direct access to the simulator command line interface

    cmd_args: list of strings where each item is a argument for the C main function
    ex: ["4","rdc2","100","-c"]
    '''
    # How to pass list of strings came from http://stackoverflow.com/a/11641327
    argv = ["BEE_SNN"] + cmd_args
    return BEE_main(len(argv), (ctypes.c_char_p*len(argv))(*argv))


def generate_connections():
    '''
    Calls the C functions to automatically generate the connections
    '''
    if BEE_initialized():
        liquid_generate_connections()
        return 0
    else:
        print "Simulator is not initialized!"
        return 1


def process_connections():
    '''
    Calls the C functions to automatically process the connections.
    If the user wants to create connections (instead of calling the generate_connections or using a pre-saved bin file), it's necessary
    to allocate memory and populate:
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

    But this is not implemented, yet, because the variables are not being exposed to Python...
    '''
    if BEE_initialized():
        liquid_process_connections()
        return 0
    else:
        print "Simulator is not initialized!"
        return 1


# Liquid's stats
def output_stats():
    '''
    Returns an numpy array with:
    - Total number of neurons
    - Number of inhibitory neurons
    - Number of excitatory neurons
    - Number of inhibitory connections
    - Number of excitatory connections
    '''
    if BEE_initialized() and BEE_connected():
        output = numpy.empty(4,dtype=numpy.int32)
        liquid_stats(output)
        print "Total number of neurons:", output[0]+output[1]
        print "Number of inhibitory neurons:",output[0]
        print "Number of excitatory neurons:",output[1]
        print "Number of inhibitory connections:",output[2]
        print "Number of excitatory connections:",output[3]
        return numpy.concatenate(([output[0]+output[1]],output),axis=0)
    else:
        print "Simulator is not ready!"


# Update the simulation (step) and returns the current time
def updates_sim(input_spikes_exc, input_spikes_inh, input_spikes_exc_w, input_spikes_inh_w, size_exc, size_inh):
    '''
    Updates the simulation (step) and returns the current simulation time
        input_spikes_exc:   array of neurons indices receiving excitatory spikes
        input_spikes_inh:   array of neurons indices receiving excitatory spikes
        input_spikes_exc_w: array with the weights for the excitatory spikes
        input_spikes_inh_w: array with the weights for the inhibitory spikes
        size_exc: size of input_spikes_exc
        size_inh: size of input_spikes_inh
    '''
    if BEE_initialized() and BEE_connected():
        input_spikes_exc = numpy.array(input_spikes_exc,dtype=numpy.int32)
        input_spikes_inh = numpy.array(input_spikes_inh,dtype=numpy.int32)
        input_spikes_exc_w = numpy.array(input_spikes_exc_w,dtype=numpy.float32)
        input_spikes_inh_w = numpy.array(input_spikes_inh_w,dtype=numpy.float32)
        size_exc = ctypes.c_int32(size_exc)
        size_inh = ctypes.c_int32(size_inh)
        return liquid_update(input_spikes_exc,input_spikes_inh,input_spikes_exc_w,input_spikes_inh_w,size_exc,size_inh)
    else:
        print "Simulator is not ready!"


# Reads the output spikes
def output_sim(number_of_neurons):
    '''
    Returns the last output spikes True/False (numpy array Bool)
    - Useful to apply directly as a filter with a numpy array
    '''
    if BEE_initialized():
        output = numpy.empty(number_of_neurons,dtype=numpy.int32)
        liquid_output(output)
        return numpy.array(output>0)
    else:
        print "Simulator is not ready!"


def output_sim_full(number_of_neurons):
    '''
    Returns the last output spikes indices including the no spiking ones (numpy array int32)
    '''
    if BEE_initialized():
        output = numpy.empty(number_of_neurons,dtype=numpy.int32)
        liquid_output(output)
        return numpy.array(output)
    else:
        print "Simulator is not ready!"


# Reads output spikes - returns an array with the indices of the neurons who spiked
def reads_spikes(number_of_neurons):
    '''
    Returns only the last output spikes indices (numpy array int32)
    '''
    if BEE_initialized():
        output = numpy.empty(number_of_neurons,dtype=numpy.int32)
        total = liquid_spikes(output)
        return output[:total]
    else:
        print "Simulator is not ready!"


# Reads the membrane voltages
def output_voltages(number_of_neurons):
    '''
    Returns the current membrane voltages (numpy array with their values).
    '''
    if BEE_initialized():
        output = numpy.empty(number_of_neurons,dtype=numpy.float32)
        liquid_voltages(output)
        return numpy.array(output)
    else:
        print "Simulator is not ready!"


# Reads the initial membrane voltages
def output_initial_voltages(number_of_neurons):
    '''
    Returns the initial value used for the membrane voltages (numpy array float32).
    '''
    if BEE_initialized():
        output = numpy.empty(number_of_neurons,dtype=numpy.float32)
        liquid_initial_voltages(output)
        return numpy.array(output)
    else:
        print "Simulator is not ready!"


# Reads the excitatory currents
def output_exc_currents(number_of_neurons):
    '''
    Returns the current values of the excitatory exponential synapses (numpy array float32).
    '''
    if BEE_initialized():
        output = numpy.empty(number_of_neurons,dtype=numpy.float32)
        liquid_exc_currents(output)
        return numpy.array(output)
    else:
        print "Simulator is not ready!"


# Reads the inhibitory currents
def output_inh_currents(number_of_neurons):
    '''
    Returns the current values of the inhibitory exponential synapses (numpy array float32).
    '''
    if BEE_initialized():
        output = numpy.empty(number_of_neurons,dtype=numpy.float32)
        liquid_inh_currents(output)
        return numpy.array(output)
    else:
        print "Simulator is not ready!"


# Reads the pre_i connections
def output_pre_i_connections(number_of_neurons):
    '''
    Returns the indices of the inhibitory=>? PRE-synaptic connections.
    '''
    if BEE_initialized() and BEE_connected():
        output = numpy.empty(number_of_neurons,dtype=numpy.int32)
        liquid_pre_i(output)
        return numpy.array(output)
    else:
        print "Simulator is not ready!"


# Reads the pos_i connections
def output_pos_i_connections(number_of_neurons):
    '''
    Returns the indices of the inhibitory=>? POS-synaptic connections.
    '''
    if BEE_initialized() and BEE_connected():
        output = numpy.empty(number_of_neurons,dtype=numpy.int32)
        liquid_pos_i(output)
        return numpy.array(output)
    else:
        print "Simulator is not ready!"


# Reads the pre_i weights
def output_pre_i_weights(number_of_neurons):
    '''
    Returns the weight values of the inhibitory=>? connections.
    '''
    if BEE_initialized() and BEE_connected():
        output = numpy.empty(number_of_neurons,dtype=numpy.float32)
        liquid_w_i(output)
        return numpy.array(output)
    else:
        print "Simulator is not ready!"


# Reads the pre_e connections
def output_pre_e_connections(number_of_neurons):
    '''
    Returns the indices of the excitatory=>? PRE-synaptic connections.
    '''
    if BEE_initialized() and BEE_connected():
        output = numpy.empty(number_of_neurons,dtype=numpy.int32)
        liquid_pre_e(output)
        return numpy.array(output)
    else:
        print "Simulator is not ready!"


# Reads the pos_e connections
def output_pos_e_connections(number_of_neurons):
    '''
    Returns the indices of the excitatory=>? POS-synaptic connections.
    '''
    if BEE_initialized() and BEE_connected():
        output = numpy.empty(number_of_neurons,dtype=numpy.int32)
        liquid_pos_e(output)
        return numpy.array(output)
    else:
        print "Simulator is not ready!"


# Reads the pre_e weights
def output_pre_e_weights(number_of_neurons):
    '''
    Returns the weight values of the excitatory=>? connections.
    '''
    if BEE_initialized() and BEE_connected():
        output = numpy.empty(number_of_neurons,dtype=numpy.float32)
        liquid_w_e(output)
        return numpy.array(output)
    else:
        print "Simulator is not ready!"


# Reads the excitatory indices
def output_exc_indices(number_of_neurons):
    '''
    Returns the array with the neuron's indices belonging to excitatory group.
    '''
    if BEE_initialized() and BEE_connected():
        output = numpy.empty(number_of_neurons,dtype=numpy.int32)
        liquid_exc_indices(output)
        return numpy.array(output)
    else:
        print "Simulator is not ready!"


# Reads the inhibitory indices
def output_inh_indices(number_of_neurons):
    '''
    Returns the array with the neuron's indices belonging to inhibitory group.
    '''
    if BEE_initialized() and BEE_connected():
        output = numpy.empty(number_of_neurons,dtype=numpy.int32)
        liquid_inh_indices(output)
        return numpy.array(output)
    else:
        print "Simulator is not ready!"


# Reads the noisy currents
def output_noisy_currents(number_of_neurons):
    '''
    Returns the current values of the noisy currents (numpy array float32).
    '''
    if BEE_initialized():
        output = numpy.empty(number_of_neurons,dtype=numpy.float32)
        liquid_noisy_currents(output)
        return numpy.array(output)
    else:
        print "Simulator is not ready!"


# Reads the constant offset currents
def output_noisy_offsets(number_of_neurons):
    '''
    Returns the values of the fixed offset noisy currents (numpy array float32).
    '''
    if BEE_initialized():
        output = numpy.empty(number_of_neurons,dtype=numpy.float32)
        liquid_noisy_offset(output)
        return numpy.array(output)
    else:
        print "Simulator is not ready!"


# Reads the refractory values
def output_refrac_values(number_of_neurons):
    '''
    Returns the values of the refractory periods (numpy array float32).
    '''
    if BEE_initialized() and BEE_connected():
        output = numpy.empty(number_of_neurons,dtype=numpy.float32)
        liquid_refrac_values(output)
        return numpy.array(output)
    else:
        print "Simulator is not ready!"


# Reads connected test
def output_connected(number_of_neurons):
    '''
    Verifies if the simulator is ready!
    '''
    if BEE_initialized() and BEE_connected():
        output = numpy.empty(number_of_neurons,dtype=numpy.int32)
        liquid_connected(output)
        return numpy.array(output)
    else:
        print "Simulator is not ready!"


# Receives spikes from the numpy.arrays (matrices) and returns the outputs
def ext_update(exc_spikes, inh_spikes, exc_weights, inh_weights, size_exc, size_inh, number_of_neurons, number_of_iterations):
    '''
    Receives spikes from the numpy.arrays (matrices) and returns the outputs.
    This function uses a for loop that runs inside the C code, therefore it should be faster than calling the updates_sim
    in a Python for loop.
    '''
    if BEE_initialized() and BEE_connected():
        output = numpy.empty((number_of_iterations,number_of_neurons),dtype=numpy.int32)
        liquid_ext_update(exc_spikes, inh_spikes, exc_weights, inh_weights, size_exc, size_inh, output, number_of_iterations)
        return output
    else:
        print "Simulator is not ready!"



def BEE_free():
    '''
    Frees all the memory allocated by the C shared library and makes possible to restart the simulator again.
    '''
    if BEE_initialized():
        BEE_freeC()
        return time.time()
    else:
        print "Simulator is not initialized!"


# coding: utf-8

# This is a VERY VERY simple Spiking Neural Network simulator that creates a Liquid State Machine
# 
# It uses the ideas from:
# Maass, Wolfgang, Thomas Natschlager, and Henry Markram. 2002. 
# “Real-Time Computing without Stable States: A New Framework for Neural Computation Based on Perturbations.” 
# Neural Computation 14 (11): 2531–60. doi:10.1162/089976602760407955.
# 
# However, the dynamic synapses (STP) and the time delays are NOT implemented (yet) in an attempt of making the system faster
# and also because I'm not sure if they are really necessary.
# 
# Probably this code has a lot of bugs because I've stopped developing it without properly testing it. Right now I'm working in a complete C version of a LSM simulator.

# In[1]:

# If you want to use Numpy, Matplotlib, etc loading, uncomment the line below:
# %pylab inline


# In[1]:

get_ipython().run_cell_magic(u'file', u'multiple_find.c', u'/* File multiple_find.c */\nvoid multiple_find(int n, int m, int *pre, int *spiked, int *output, int *l) {\n   //n=>length of pre\n   //m=>length of spiked\n  int i;\n  int j;\n  int k=0;\n\n  for (i=0;i<n;i++) {\n      for (j=0;j<m;j++){\n        if (pre[i]==spiked[j]){\n            output[k] = i;\n            k++;\n        }\n      }\n  }\n  *l=k;\n}')


# In[2]:

get_ipython().run_cell_magic(u'file', u'snnsim_c.pyf', u'! File snnsim_c.pyf\npython module snnsim_c\ninterface\n  subroutine multiple_find(n,m,pre,spiked,output,*l)\n    intent(c) multiple_find                 ! multiple_find is a C function\n    intent(c)                               ! all multiple_find arguments are \n                                            ! considered as C based\n    integer intent(hide), depend(pre) :: n=len(pre)  ! n is the length\n                                                     ! of input array pre\n    \n    integer intent(hide), depend(spiked) :: m=len(spiked)  ! m is the length\n                                                           ! of input array spiked\n\n    integer intent(in) :: pre(n)       ! pre is input array \n    \n    integer intent(in) :: spiked(m)    ! spiked is input array  \n\n    integer intent(out) :: output(n)   ! output is output array\n\n    integer intent(out) :: l                                                  \n\n  end subroutine multiple_find\nend interface\nend python module snnsim_c')


# In[5]:

# Uses the f2py to generate the Python module!
# This program SHOULD be installed automatically with Numpy.
# http://docs.scipy.org/doc/numpy-dev/f2py/
# REMEMBER: iPython will NOT reload a module after being imported once. It is necessary to restart the kernel!!!!
get_ipython().system(u'f2py snnsim_c.pyf multiple_find.c -c >> null')
# !f2py snnsim_c.pyf multiple_find.c -c


# In[6]:

get_ipython().run_cell_magic(u'writefile', u'liquid_structure.c', u'\n#include <stdlib.h>\n#include <stdio.h>\n#include <math.h>\n\n\n/*\nCompile as a shared library:\ngcc -shared -Wl,-install_name,liquid_structure.so -o liquid_structure.so -fPIC liquid_structure.c -O2\n*/\n\n// This is used to define when realloc is going to ask for more memory\n#define MAX_CONNECTIONS 10\n\n// These are the addresses we need to free sometime in the future :)\n// They are global variables to make easier to access from two different functions\nint *f_ptr_pre_i = NULL; \nint *f_ptr_pos_i = NULL; \nint *f_ptr_pre_e = NULL;\nint *f_ptr_pos_e = NULL;\ndouble *f_ptr_w_i = NULL;\ndouble *f_ptr_w_e = NULL;\n    \ndouble gaussrand(unsigned int my_seed){\n    /*\n    Gaussian generator algorithm from: \n            http://c-faq.com/lib/gaussian.html\n            Marsaglia and Bray, "A Convenient Method for Generating Normal Variables"\n            http://en.wikipedia.org/wiki/Marsaglia_polar_method\n            Abramowitz and Stegun, Handbook of Mathematical Functions\n    Expect value: 0\n    Standard deviation: 1\n    */\n\n    static unsigned int seed;\n    static double V1, V2, S;\n    static int phase = 0;\n    double X;\n\n    //Initializes the gaussian random generator\n    if(my_seed!=0){\n        seed = my_seed;\n        phase = 0;\n    }\n\n    if(phase == 0) {\n        do {\n            double U1 = (double)rand_r(&seed) / RAND_MAX;\n            double U2 = (double)rand_r(&seed) / RAND_MAX;\n\n            V1 = 2 * U1 - 1;\n            V2 = 2 * U2 - 1;\n            S = V1 * V1 + V2 * V2;\n            } while(S >= 1 || S == 0);\n\n        X = V1 * sqrt(-2 * log(S) / S);\n    } else\n        X = V2 * sqrt(-2 * log(S) / S);\n\n    phase = 1 - phase;\n\n    return X;\n}\n\n\ndouble euclidean_dist(int i, int j, int xd, int yd){\n    // I\'m assuming that in the 3D structure the values i and j are equivalent \n    // to the incrementing of x (until complete the line) then y and (after complete xy plane) z values.\n    // xd, yd are the x and y dimensions of the liquid (shape)\n    double xy = xd*yd;\n\n    double zi = (int) (i/xy); \n    double zj = (int) (j/xy); \n\n    int yi = (int) (i-zi*xy)/xd; \n    int yj = (int) (j-zj*xy)/xd;\n\n    double xi = (int) (i-yi*zi*xy); \n    double xj = (int) (j-yj*zj*xy); \n\n    // It\'s necessary to centralize the axis, otherwise we are going to have a bias\n    xi = -xi/2.0;   // Centralize in the x=0 axis\n    xj = -xj/2.0;   // Centralize in the x=0 axis \n\n    zi = -zi/2.0;   // Centralize in the z=0 axis\n    zj = -zj/2.0;   // Centralize in the z=0 axis    \n\n    return sqrt(pow((xi-xj),2)+pow((yi-yj),2)+pow((zi-zj),2));\n}\n\n\n//Frees the memory space allocated to the liquid structure\nint free_variables(void){\n    static int memory_used = 1;\n\n    if(memory_used!=0){\n        //Deallocate the memory\n        free(f_ptr_pre_i); \n        free(f_ptr_pos_i); \n        free(f_ptr_pre_e);\n        free(f_ptr_pos_e);\n        free(f_ptr_w_i);\n        free(f_ptr_w_e);\n        //Signalize the memory was freed\n        memory_used = 0;\n        return 0;\n    }else{\n        return 1;\n    }\n\n}\n\nint generate_connections(int x, \n                         int y, \n                         int z, \n                         double lbd, \n                         int *inh_indices, int inh_size, \n                         double parametres[2][2][6], \n                         int *inh_connections,\n                         int **ptr_pre_i, int **ptr_pos_i, double **ptr_w_i, \n                         int *exc_connections,                         \n                         int **ptr_pre_e, int **ptr_pos_e, double **ptr_w_e, \n                         unsigned int my_seed){\n\n    // inh_indices MUST be a list made of UNIQUE indices (integers)!\n\n    unsigned int c_seed = my_seed;\n    int number_of_neurons = x*y*z;\n    int total_number_of_connections = 0; //counts how many connecitons are created\n    \n    int inh_c = 0; //counts how many inhibitory=>? connections\n    int exc_c = 0; //counts how many excitatory=>? connections\n\n    int PRE; //flags to indicate the type of connection: inhibitory=>0 and excitatory=>1\n    int POS;\n\n    double temp = gaussrand(my_seed); //initializes the random generator\n\n    int n_malloc_i = 1; //Used with realloc to dynamically allocate more memory: inhibitory connections\n    int n_malloc_e = 1; //Used with realloc to dynamically allocate more memory: excitatory connections\n\n    int *pre_i = NULL;\n    int *pos_i = NULL;\n    double *w_i = NULL;\n\n    int *pre_e = NULL;\n    int *pos_e = NULL;\n    double *w_e = NULL;\n\n    //Allocate memory to save the inhibitory=>? connections\n    pre_i = calloc(MAX_CONNECTIONS,sizeof(int));\n    pos_i = calloc(MAX_CONNECTIONS,sizeof(int));\n    w_i = calloc(MAX_CONNECTIONS,sizeof(double));\n\n    //Allocate memory to save the excitatory=>? connections\n    pre_e = calloc(MAX_CONNECTIONS,sizeof(int));\n    pos_e = calloc(MAX_CONNECTIONS,sizeof(int));\n    w_e = calloc(MAX_CONNECTIONS,sizeof(double));\n\n    if (pre_i==NULL || pos_i==NULL || w_i==NULL || pre_e==NULL || pos_e==NULL || w_e==NULL){\n        fprintf(stderr,"connections memory malloc ERROR!");\n        exit(1); //I need to check which error I should signal here...\n    }\n\n\n    // Goes through all the neurons indices and verify if they are inhbitory (excitatory test is implicit)\n    for (int i=0;i<number_of_neurons; i++){ //These are the PRE indices\n        PRE = 1; //Because it\'s only tested against inhbitory indices\n\n        // Verifies if the neuron index \'i\' belongs to inh_indices\n        for (int k=0; k<inh_size; k++){\n            if(inh_indices[k]==i){// Tests if the connection is I?\n                PRE = 0; //Indicates the PRE-synaptic neuron is inhbitory\n                break; //Saves CPU cycles...\n            }\n        }\n\n\n        for (int j=0;j<number_of_neurons; j++){ //These are the POS indices\n            POS = 1;//Because it\'s only tested against inhbitory indices\n            \n            // Verifies if the neurons indices \'i\' and \'j\' belongs to inh_indices\n            for (int k=0; k<inh_size; k++){\n                if(inh_indices[k]==j){// Tests if the connection is I?\n                    POS = 0; //Indicates the POS-synaptic neuron is inhbitory\n                    break; //Saves CPU cycles...\n                }\n            }\n            \n            //Here the variables about PRE/POS are set\n\n            // Verifies if a connection (synapse) will be created or not depending on the calculated probability.\n            if ( ((double)rand_r(&c_seed)/(double)RAND_MAX) <= (parametres[PRE][POS][0]*exp(-pow(euclidean_dist(i,j,x,y)/lbd,2))) ){\n                //It means we have a synapse here!\n                total_number_of_connections++;                \n\n                if (PRE==0){//Inhibitory=>? connection\n                    inh_c++;\n                    \n                    //Verifies if the buffer (used to store the Inhibitory=>? connections) needs more memory\n                    if (inh_c>(MAX_CONNECTIONS*n_malloc_i)){\n                        n_malloc_i++;\n                        pre_i = realloc(pre_i,n_malloc_i*MAX_CONNECTIONS*sizeof(int));\n                        pos_i = realloc(pos_i,n_malloc_i*MAX_CONNECTIONS*sizeof(int));\n                        w_i = realloc(w_i,n_malloc_i*MAX_CONNECTIONS*sizeof(double));\n                        if (pre_i==NULL || pos_i==NULL || w_i==NULL){\n                            fprintf(stderr,"connections memory realloc ERROR!");\n                            exit(1);\n                        }\n                    }\n                         \n                    pre_i[inh_c-1]=i;\n                    pos_i[inh_c-1]=j;\n                    w_i[inh_c-1]=1E-9*parametres[PRE][POS][4]*fabs(gaussrand(0)*0.5+1); //AMaass\n\n                }else{//Excitatory=>? connections\n                    exc_c++;   \n                    \n                    //Verifies if the buffer (used to store the Inhibitory=>? connections) needs more memory\n                    if (exc_c>(MAX_CONNECTIONS*n_malloc_e)){\n                        n_malloc_e++;\n                        pre_e = realloc(pre_e,n_malloc_e*MAX_CONNECTIONS*sizeof(int));\n                        pos_e = realloc(pos_e,n_malloc_e*MAX_CONNECTIONS*sizeof(int));\n                        w_e = realloc(w_e,n_malloc_e*MAX_CONNECTIONS*sizeof(double));\n                        if (pre_e==NULL || pos_e==NULL || w_e==NULL){\n                            fprintf(stderr,"connections memory realloc ERROR!");\n                            exit(1);\n                        }\n                    }\n                \n                    pre_e[exc_c-1]=i;\n                    pos_e[exc_c-1]=j;\n                    w_e[exc_c-1]=1E-9*parametres[PRE][POS][4]*fabs(gaussrand(0)*0.5+1); //AMaass\n                }\n            }\n\n        }\n    }\n\n\n    *ptr_pre_i = pre_i;\n    *ptr_pos_i = pos_i;\n    *ptr_w_i = w_i;\n    *inh_connections = inh_c;\n\n    *ptr_pre_e = pre_e;\n    *ptr_pos_e = pos_e;\n    *ptr_w_e = w_e;\n    *exc_connections = exc_c;\n\n    //Saves the memory addresses to be freed later\n    f_ptr_pre_i = pre_i; \n    f_ptr_pos_i = pos_i; \n    f_ptr_pre_e = pre_e;\n    f_ptr_pos_e = pos_e;\n    f_ptr_w_i = w_i;\n    f_ptr_w_e = w_e;\n\n    return total_number_of_connections;\n}')


# In[7]:

# Verbose version:
# !gcc -shared -Wl,-install_name,liquid_structure.so -o liquid_structure.so -fPIC liquid_structure.c -O2 -v

# No messages version:
get_ipython().system(u'gcc -shared -Wl,-install_name,liquid_structure.so -o liquid_structure.so -fPIC liquid_structure.c -O2 >> null')


# In[1]:

get_ipython().run_cell_magic(u'file', u'LSM_simulator.py', u'\n\'\'\'\nUSING EXPONENTIAL INPUT CURRENTS\n\'\'\'\nimport numpy\n\nimport sys\n\n\n\nimport snnsim_c # C compiled multiple search algorithm inside numpy arrays\nreload(sys.modules[\'snnsim_c\']) \n\n# from scipy.integrate import odeint # I\'m not using because I\'m doing the numerical integration by hand, but it is there if I need to test\n\n# C library used for the liquid structure generation\nimport ctypes\nliquid_structure=ctypes.CDLL("./liquid_structure.so")\n\n\nclass SpikingLiquid(object):\n    \n    def reset(self):\n        \'\'\'\n        Resets the initial values of the simulation, but keeps the internal structure of the liquid\n        \'\'\'\n        # Generates individual RandomState objects for each stochastic thing in the code\n        self.rndste=[numpy.random.RandomState(self.seeds[i] & 0xFFFFFFFF) if self.seeds[i]!=None else numpy.random.RandomState() for i in range(5)]\n        # RANDOM-0: Membrane initial potentials\n        # RANDOM-1: Noisy offset currents\n        # RANDOM-2: Selection of the inhibitory and excitatory neurons\n        # RANDOM-3: Internal connections of the liquid\n        # RANDOM-4: Noisy corrents\n        #\n        \n        \n        # RANDOM-0\n        # Membrane initial potentials\n        \n        # Initializes the membrane potentials for all neurons randomly according to (Maass, Natschlager, and Markram 2002)\n        self.neurons_membrane = self.rndste[0].uniform(13.5E-3,15E-3, self.Number_of_Neurons) #in mV \n\n        self.neurons_exc_curr = numpy.zeros(self.Number_of_Neurons) # Initializes the excitatory currents levels for all neurons\n        self.neurons_inh_curr = numpy.zeros(self.Number_of_Neurons) # Initializes the inhibitory currents levels for all neurons\n        \n        self.time = 0 # Starts the simulation at time=0\n        self.times_exc_curr = numpy.zeros(self.Number_of_Neurons) # Initializes the times for the excitatory currents\n        \n        self.refrac_timer = numpy.zeros(self.Number_of_Neurons) # Initializes the refractory timers\n        \n        self.test_vrest  = numpy.zeros(self.Number_of_Neurons)>0 # Initializes all positions with False\n        self.test_vthres = numpy.zeros(self.Number_of_Neurons)>0 # Initializes all positions with False\n        self.test_refrac = numpy.zeros(self.Number_of_Neurons)>0 # Initializes all positions with False\n        \n\n        # RANDOM-1\n        # Noisy offset currents\n        \n        # These are the offset noisy currents according to (Maass, Natschlager, and Markram 2002)\n        self.noisy_offset_currents = self.rndste[1].uniform(14.975E-9,15.025E-9, self.Number_of_Neurons) # in nA\n    \n    \n    def __init__(self, Number_of_Neurons=135, Net_shape=(15,3,3), lbd_value=1.2, step=0.2E-3, taum=30E-3, cm=30E-9, vrest=(0.0)*numpy.ones(135), vthres=(15E-3)*numpy.ones(135), vresets=numpy.ones(135)*13.5E-3, taue=3E-3, taui=6E-3, refractory=(3E-3,2E-3), seeds=(None,None,None,None,None)):\n        \'\'\'\n        Creates a liquid state machine.\n        \n        Number_of_Neurons: total number of neurons to be simulated\n        Net_shape: a tuple with the liquid\'s 3D shape (x,y,z)\n        step: simulation time step (in seconds)\n        \n        lbd_value: coefficient of the liquid\'s connections (see Maass, Natschlager, and Markram 2002)\n        \n        taum: membrane time constant (in seconds)\n        cm: membrane capacitance (in farads)\n        \n        vrest: array with the resting membrane potential (in volts)\n        vthres: array with the membrane spiking threshold (in volts)\n        vresets: array with the reset voltages of all neurons (in volts)        \n        \n        taue: current decay time constant excitatory (in seconds)\n        taui: current decay time constant inhibitory (in seconds)\n        \n        refractory: tuple where the elements tell the refractory for excitatory and inhibitory neurons (in seconds)\n        \n        seeds: tuple with five values to be used as random seed to the RandomState objects. If None is passed to one\n        of the tuple values that RandomState object will keep as initialized with .RandomState().\n            seeds[0]: Membrane initial potentials\n            seeds[1]: Noisy offset currents\n            seeds[2]: Selection of the inhibitory and excitatory neurons\n            seeds[3]: Internal connections of the liquid\n            seeds[4]: Noisy corrents\n        \n        To run the simulation call:\n        .update(spikes_exc, spikes_inh, weights_exc, weights_inh):\n        spikes_exc: list/array with the index of the neurons receiving excitatory spikes\n        spikes_inh: list/array with the index of the neurons receiving inhibitory spikes\n        weights_exc: list/array with the weights in the same order as spikes_exc\n        weights_inh: list/array with the weights in the same order as spikes_inh\n        \n        To verify the membrane voltages call:\n        .verify_membrane()\n\n        To verify which neuron spiked call:        \n        .verify_spikes()\n        \'\'\'\n        print "Initializing the liquid..."\n        print "Number of neurons in the liquid: ", Number_of_Neurons\n        print "Liquid\'s shape: ",Net_shape\n        print "Simulation time step: ",step\n        \n        self.seeds = seeds\n                \n        self.Number_of_Neurons = Number_of_Neurons\n        \n        self.step = step\n        \n        self.cm = cm\n        self.vrest = numpy.array(vrest) # Guarantees it is a Numpy.Array\n        self.vthres = numpy.array(vthres)\n        self.vresets = numpy.array(vresets)\n                \n        self.taum = taum\n        self.taue = taue\n        self.taui = taui\n\n        self.reset() # Initializes the internal variables to start the simulation\n        \n        # RANDOM-2\n        # Selection of the inhibitory and excitatory neurons\n\n        # Selects some random neurons to be inhibitory/excitatory ones\n        lsm_3dGrid_flat = numpy.arange(self.Number_of_Neurons) # Creates a dummy array to be used with the suffle        \n        self.rndste[2].shuffle(lsm_3dGrid_flat) # Using the shuffle method I have guarantee there are NO repetitions\n        \n        self.inhbitory_index = numpy.array(lsm_3dGrid_flat[:int(self.Number_of_Neurons*0.2)], dtype=numpy.int32) # The first (approx.) 20% as inhibitory\n        self.excitatory_index = numpy.array(lsm_3dGrid_flat[int(self.Number_of_Neurons*0.2):])# The rest of the neurons as excitatory\n\n        self.refractory_vector = numpy.zeros(self.Number_of_Neurons)\n        self.refractory_vector[self.excitatory_index] = refractory[0] # Sets the refractory periods according to the type of neuron\n        self.refractory_vector[self.inhbitory_index] = refractory[1]  # using the information from the user input tuple.\n        \n        # This is the dictionary that has all the connections parameters according to (Maass, Natschlager, and Markram 2002).\n        # It is necessary to create the 3D connections and the STP configuration matrixes (STP is not implemented )\n        # E=>1 (excitatory) and I=>0 (inhibitory)\n        # Ex.: (0,0) => II\n        # Dynamical Synapses Parameters (STP/Liquid Structure):\n        Connections_Parameters = numpy.array([\n             [\n                numpy.array(\n                                [ # II [0,0]\n                                      0.1,       # CGupta=0.1        # Parameter used at the connection probability - from Maass2002 paper\n                                      0.32,      # UMarkram=0.32     # Use (U) - Parameter used at the Dynamic Synapse - from Maass2002 paper\n                                      0.144,     # DMarkram=0.144    # Time constant for Depression (tau_rec) - used at the Dynamic Synapse - from Maass2002 paper                    \n                                      0.06,      # FMarkram=0.06     # Time constant for Facilitation (tau_facil) - used at the Dynamic Synapse - from Maass2002 paper\n                                      -2.8,      # AMaass=-2.8        # (nA) In the Maass2002 paper\n                                      0.8        # Delay_trans = 0.8 # In Maass paper the transmission delay is 0.8 to II, IE and EI        \n                                ],dtype=numpy.double\n                            ),\n                numpy.array(\n                                [ # IE [0,1]\n                                      0.4,    # CGupta=0.4\n                                      0.25,   # UMarkram=0.25\n                                      0.7,    # DMarkram=0.7\n                                      0.02,   # FMarkram=0.02\n                                      -3.0,   # AMaass=-3.0\n                                      0.8     # Delay_trans = 0.8 # in Maass paper the transmission delay is 0.8 to II, IE and EI\n                                  ],dtype=numpy.double\n                            )\n            ],\n            [\n                numpy.array(\n                                [ # EI [1,0]\n                                      0.2,    # CGupta=0.2\n                                      0.05,   # UMarkram=0.05\n                                      0.125,  # DMarkram=0.125\n                                      1.2,    # FMarkram=1.2\n                                      1.6,    # AMaass=1.6\n                                      0.8     # Delay_trans = 0.8 # in Maass paper the transmission delay is 0.8 to II, IE and EI\n                                  ],dtype=numpy.double\n                            ),\n                numpy.array(\n                                [ # EE [1,1]\n                                      0.3,    # CGupta=0.3 \n                                      0.5,    # UMarkram=0.5\n                                      1.1,    # DMarkram=1.1\n                                      0.05,   # FMarkram=0.05\n                                      1.2,    # AMaass=1.2 #scaling parameter or absolute synaptic efficacy or weight - from Maass2002 paper\n                                      1.5     # Delay_trans = 1.5 # in Maass paper the transmission delay is 1.5 to EE connection\n                                  ],dtype=numpy.double\n                            )\n            ]\n        ])\n\n        # RANDOM-3 - self.seeds[3]\n        # Internal connections of the liquid\n        ptr2int=ctypes.POINTER(ctypes.c_int32)\n        ptr2double=ctypes.POINTER(ctypes.c_double)\n\n        gen_conn=liquid_structure.generate_connections\n        gen_conn.restype=ctypes.c_int32\n        gen_conn.argtypes = [\n                        ctypes.c_int32, \n                        ctypes.c_int32, \n                        ctypes.c_int32, \n                        ctypes.c_double,\n                        numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags=\'ALIGNED,C_CONTIGUOUS\'), ctypes.c_int32,\n                        numpy.ctypeslib.ndpointer(dtype=numpy.double, flags=\'ALIGNED,C_CONTIGUOUS\'),\n                        ctypes.POINTER(ctypes.c_int32),\n                        ctypes.POINTER(ptr2int), ctypes.POINTER(ptr2int), ctypes.POINTER(ptr2double),\n                        ctypes.POINTER(ctypes.c_int32),\n                        ctypes.POINTER(ptr2int), ctypes.POINTER(ptr2int), ctypes.POINTER(ptr2double),\n                        ctypes.c_uint32\n                       ]\n        # int generate_connections(int x, \n        #                          int y, \n        #                          int z, \n        #                          double lbd, \n        #                          long int *inh_indices, int inh_size, \n        #                          double parametres[2][2][6], \n        #                          int *inh_connections,\n        #                          int **ptr_pre_i, int **ptr_pos_i, double **ptr_w_i, \n        #                          int *exc_connections,                         \n        #                          int **ptr_pre_e, int **ptr_pos_e, double **ptr_w_e, \n        #                          unsigned int my_seed)\n\n        number_of_inh_neurons = ctypes.c_int(len(self.inhbitory_index))\n\n        inh_connections = ctypes.c_int32(0)\n        p_pre_i = ptr2int()\n        p_pos_i = ptr2int()\n        p_w_i = ptr2double()\n\n        exc_connections = ctypes.c_int32(0)\n        p_pre_e = ptr2int()\n        p_pos_e = ptr2int()\n        p_w_e = ptr2double()\n\n        seed = ctypes.c_uint32(123422)\n\n        total_number_of_connections = 0;\n\n        # int generate_connections(int x, \n        #                          int y, \n        #                          int z, \n        #                          double lbd, \n        #                          long int *inh_indices, int inh_size, \n        #                          double parametres[2][2][6], \n        #                          int *inh_connections,\n        #                          int **ptr_pre_i, int **ptr_pos_i, double **ptr_w_i, \n        #                          int *exc_connections,                         \n        #                          int **ptr_pre_e, int **ptr_pos_e, double **ptr_w_e, \n        #                          unsigned int my_seed)\n        total_number_of_connections = gen_conn(ctypes.c_int32(Net_shape[0]),\n                                               ctypes.c_int32(Net_shape[1]),\n                                               ctypes.c_int32(Net_shape[2]),\n                                               ctypes.c_double(lbd_value),\n                                               self.inhbitory_index, number_of_inh_neurons,\n                                               Connections_Parameters,\n                                               ctypes.byref(inh_connections),\n                                               ctypes.byref(p_pre_i), ctypes.byref(p_pos_i), ctypes.byref(p_w_i),\n                                               ctypes.byref(exc_connections),\n                                               ctypes.byref(p_pre_e), ctypes.byref(p_pos_e), ctypes.byref(p_w_e),\n                                               ctypes.c_uint(self.seeds[3])\n                                              )\n        \n        # Processes the information about the liquid=>liquid connections generated with the lm.generate_connections\n        # The lists comprehensions below are processing the neurons that HAVE a connection. Because it is a probabilist function\n        # some neurons will have more than one connection, but some can have none!\n        # These connections will be used to verify and inject spikes at each time step during the simulation.\n        self.pre_i = numpy.ctypeslib.as_array((ctypes.c_int32 * inh_connections.value).from_address(ctypes.addressof(p_pre_i.contents)))\n        self.pos_i = numpy.ctypeslib.as_array((ctypes.c_int32 * inh_connections.value).from_address(ctypes.addressof(p_pos_i.contents)))\n        self.w_i = numpy.ctypeslib.as_array((ctypes.c_double * inh_connections.value).from_address(ctypes.addressof(p_w_i.contents)))\n        \n#         self.pre_i = numpy.array([ind[0][0] for ind in self.output_L_L[\'inh\']],dtype=int) # Extracts the indices of the PRE neurons (inhibitory)\n#         self.pos_i = numpy.array([ind[0][1] for ind in self.output_L_L[\'inh\']],dtype=int) # Extracts the indices of the POS neurons (inhibitory)\n#         self.w_i   = numpy.array([ind[2][0]*1E-9 for ind in self.output_L_L[\'inh\']],dtype=float) # Extracts the weights connecting PRE to POS (inhibitory)\n\n        self.pre_e = numpy.ctypeslib.as_array((ctypes.c_int32 * exc_connections.value).from_address(ctypes.addressof(p_pre_e.contents)))\n        self.pos_e = numpy.ctypeslib.as_array((ctypes.c_int32 * exc_connections.value).from_address(ctypes.addressof(p_pos_e.contents)))\n        self.w_e = numpy.ctypeslib.as_array((ctypes.c_double * exc_connections.value).from_address(ctypes.addressof(p_w_e.contents)))\n\n#         self.pre_e = numpy.array([ind[0][0] for ind in self.output_L_L[\'exc\']],dtype=int) # Extracts the indices of the PRE neurons (excitatory)\n#         self.pos_e = numpy.array([ind[0][1] for ind in self.output_L_L[\'exc\']],dtype=int) # Extracts the indices of the POS neurons (excitatory)\n#         self.w_e   = numpy.array([ind[2][0]*1E-9 for ind in self.output_L_L[\'exc\']],dtype=float) # Extracts the weights connecting PRE to POS (excitatory)\n\n        # pre_i, pos_i and w_i have the same length\n        # pre_e, pos_e and w_e have the same length\n        \n        print "Number of connections INH=>: ", str(len(self.w_i))\n        print "Number of connections EXC=>: ", str(len(self.w_e))\n        \n        print "Initializing the liquid...Done!"\n    \n    \n    # Based on 2.4 Synapses - http://icwww.epfl.ch/~gerstner/SPNM/node16.html, \n    # but using the same model to inhibitory and excitatory (with differents time constants)\n    def process_exc_spikes(self, spikes, weights):\n        \'\'\'\n        Processes the received spikes at the current time updating the current values.\n        spikes: list with the indexes of the neurons who spiked.\n        weights: the weights to the neurons who spiked\n        \'\'\'\n        self.neurons_exc_curr += (-self.neurons_exc_curr/self.taue)*self.step # Calculates the current values\n        self.neurons_exc_curr[spikes] +=  weights*numpy.ones(len(spikes)) # Adds to the ones that received spikes\n        \n    def process_inh_spikes(self, spikes, weights):\n        \'\'\'\n        Processes the received spikes at the current time updating the current values.\n        spikes: list with the indexes of the neurons who spiked.\n        weights: the weights to the neurons who spiked        \n        \'\'\'\n        self.neurons_inh_curr += (-self.neurons_inh_curr/self.taui)*self.step # Calculates the current values\n        self.neurons_inh_curr[spikes] += weights*numpy.ones(len(spikes)) # Adds to the ones that received spikes\n\n    def dv(self,v,t,ie,ii,i_offset,i_noise,vrest):\n        \'\'\'\n        Differential equation describing the behaviour of the neuron model\n        dv(t)/dt = (ie(t) + ii(t) + i_offset + i_noise(t))/self.cm + (vrest-v(t))/self.taum \n        \'\'\'\n        return (ie + ii + i_offset + i_noise)/self.cm + (vrest-v)/self.taum\n\n    def check_internal_spikes(self):\n        \'\'\'\n        Tests if the liquid\'s neurons spiked\n        \'\'\'\n        # Checks which internal inhibitory neurons spiked\n        spiked_i = self.inhbitory_index[self.test_vthres[self.inhbitory_index]] # Gives the index of the inh spiked ones\n        # The next line (Pure Python Version), probably, is the bottleneck of this method\n#         who_receives_spikes_i_idx = [idx for idx in xrange(len(self.pre_i)) if self.pre_i[idx] in spiked_i] # Verifies if the spiked one bellongs to the connected ones\n        # Replace by the C compiled version (below):\n        who_receives_spikes_i_idx,l = snnsim_c.multiple_find(self.pre_i,spiked_i)\n        \n        self.who_receives_spikes_i = self.pos_i[who_receives_spikes_i_idx[:l]] # Finds to who it is connected\n        self.who_receives_spikes_i_w = self.w_i[who_receives_spikes_i_idx[:l]] # Finds the connection weights\n\n        # Checks which internal excitatory neurons spiked\n        spiked_e = self.excitatory_index[self.test_vthres[self.excitatory_index]] # Gives the index of the exc spiked ones\n        # The next line (Pure Python Version), probably, is the bottleneck of this method\n#         who_receives_spikes_e_idx = [idx for idx in xrange(len(self.pre_e)) if self.pre_e[idx] in spiked_e]\n        # Replace by the C compiled version (below):\n        who_receives_spikes_e_idx,l = snnsim_c.multiple_find(self.pre_e,spiked_e)\n\n        self.who_receives_spikes_e = self.pos_e[who_receives_spikes_e_idx[:l]]\n        self.who_receives_spikes_e_w = self.w_e[who_receives_spikes_e_idx[:l]]\n        \n        \n    def verify_membrane(self):\n        \'\'\'\n        Returns a tuple: time,neurons_membrane\n        time: current simulation time (in seconds)\n        neurons_membrane: numpy.array with voltage values of the neurons\' membrane (in volts)\n        \'\'\'\n        return self.time,self.neurons_membrane\n\n    def verify_spikes(self):\n        \'\'\'\n        Returns a tuple: time, test_vthres \n        time: current simulation time (in seconds)\n        test_vthres: numpy.array of booleans indicating which neuron spiked\n        Ex:\n        indices = numpy.arange(total_number_of_neurons)\n        indices[test_vthres] => gives the indices of the neurons who spiked\n        \'\'\'\n        return self.time,self.test_vthres\n        \n    def update(self, spikes_exc, spikes_inh, weights_exc, weights_inh):\n        \'\'\'\n        Updates the simulation, running one time step, and processes the received excitatory and inhibitory spikes.\n        .update(spikes_exc, spikes_inh, weights_exc, weights_inh):\n        spikes_exc: list/array with the index of the neurons receiving excitatory spikes\n        spikes_inh: list/array with the index of the neurons receiving inhibitory spikes\n        weights_exc: list/array with the weights in the same order as spikes_exc\n        weights_inh: list/array with the weights in the same order as spikes_inh        \n        \n        The first update is time=0s\n        \'\'\'\n        self.check_internal_spikes() # runs the C compiled function to check which neuron is going to spike and returns the index of the POS neuron connected to it.\n        \n        # Here I concatenate the received spikes from outside with the internally generated spikes\n        # and update the excitatory and inhibitory currents.\n        # I used .astype(int) because numpy generates floats if one list is empty and was cheaper than testing for an empty list\n        self.process_inh_spikes(numpy.concatenate((spikes_inh,self.who_receives_spikes_i)).astype(int), numpy.concatenate((weights_inh,self.who_receives_spikes_i_w))) # Processes the inhibitories arriving spikes\n        self.process_exc_spikes(numpy.concatenate((spikes_exc,self.who_receives_spikes_e)).astype(int), numpy.concatenate((weights_exc,self.who_receives_spikes_e_w))) # Processes the excitatories arriving spikes\n\n        # RANDOM-4\n        # Noisy corrents\n        self.noisy_currents = self.rndste[4].normal(loc=0, scale=0.2E-9,size=self.Number_of_Neurons) # Variable noisy currents\n        \n        # After the currents, it is necessary to update the membrane voltage\n        # Integrating the function: dv(self,v,t,ie,ii,i_offset,i_noise,vrest)\n        \n        # Manually doing the integration:\n        self.neurons_membrane+=self.dv(self.neurons_membrane,self.time,self.neurons_exc_curr,self.neurons_inh_curr,self.noisy_offset_currents,self.noisy_currents,self.vrest)*self.step        \n\n        \n        self.test_refrac = self.refrac_timer>0 # Verifies which ones are still in the refractory period\n        self.neurons_membrane[self.test_refrac] = self.vresets[self.test_refrac] # Resets the membrane of the refractory ones\n        \n        self.test_vthres = self.neurons_membrane>self.vthres # Verifies who should spike and creates a boolean vector with this information\n\n\n        self.neurons_membrane[self.test_vthres] = self.vresets[self.test_vthres] # Resets the membrane of the spiked ones\n\n        self.refrac_timer[self.test_vthres] = self.refractory_vector[self.test_vthres] # Resets the refractory timer to the ones who spiked\n\n        # If I update the time here makes it possible to generate things at time zero!\n        self.time = self.time + self.step # Advances the simulation time one step\n        self.refrac_timer -= self.step # Subtracts the refractory timer to enable the refractory calculation\n\n    def expose_exc_currents(self):\n        return self.neurons_exc_curr\n    \n    def expose_inh_currents(self):\n        return self.neurons_inh_curr    ')


### Testing the simulator

# In[9]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# Makes the figures in the PNG format:
# For more information see %config InlineBackend
get_ipython().magic(u"config InlineBackend.figure_formats=set([u'png'])")

plt.rcParams['figure.figsize'] = 10, 6


# In[3]:

import numpy
import sys

import LSM_simulator
reload(sys.modules['LSM_simulator']) # Makes sure the interpreter is going to reload the module


# In[4]:

x,y,z = (20,5,6)
NofN = x*y*z

my_seed=1234

NofN=135
x,y,z=(15,3,3)

MaassTest=LSM_simulator.SpikingLiquid(Number_of_Neurons=NofN, Net_shape=(x,y,z), seeds=[my_seed]*5)


# In[5]:

# Initialize the arrays with the simulated values
# and saves the values for t=0s
MaassTest.reset()
t = []
spikes_idx = numpy.arange(NofN)
spikes = []
tstp,spks = MaassTest.verify_spikes()
t.append(tstp)
spikes.append(spikes_idx[spks])


# In[127]:

# Injects some spikes into the liquid!
exc_inputs = numpy.random.randint(0,NofN,size=int(NofN/2.0))
inh_inputs = numpy.random.randint(0,NofN,size=int(NofN/2.0))
exc_weights = numpy.random.rand(int(NofN/2.0))*100E-9
inh_weights = -numpy.random.rand(int(NofN/2.0))*100E-9
MaassTest.update(exc_inputs, inh_inputs, exc_weights, inh_weights)

# Appends to the output
tstp,spks = MaassTest.verify_spikes()
t.append(tstp)
spikes.append(spikes_idx[spks])


# In[151]:

for i in range(50):
    # Runs one step and save the result
    MaassTest.update([], [], [], [])

    # Appends to the output
    tstp,spks = MaassTest.verify_spikes()
    t.append(tstp)
    spikes.append(spikes_idx[spks])


# In[152]:

AINP=100

# Injects some spikes into the liquid!
exc_inputs = range(50)
inh_inputs = []
exc_weights = [AINP*1E-9]*50
inh_weights = []
MaassTest.update(exc_inputs, inh_inputs, exc_weights, inh_weights)

# Appends to the output
tstp,spks = MaassTest.verify_spikes()
t.append(tstp)
spikes.append(spikes_idx[spks])


# In[153]:

for i in range(450):
    # Runs one step and save the result
    MaassTest.update([], [], [], [])

    # Appends to the output
    tstp,spks = MaassTest.verify_spikes()
    t.append(tstp)
    spikes.append(spikes_idx[spks])


# In[6]:

def compare_to_brian():
    # Initialize the arrays with the simulated values
    # and saves the values for t=0s
    t = []
    spikes_idx = numpy.arange(NofN)
    spikes = []
    tstp,spks = MaassTest.verify_spikes()
    t.append(tstp)
    spikes.append(spikes_idx[spks])
    for i in range(50):
        # Runs one step and save the result
        MaassTest.update([], [], [], [])

        # Appends to the output
        tstp,spks = MaassTest.verify_spikes()
        t.append(tstp)
        spikes.append(spikes_idx[spks])

    AINP=100

    # Injects some spikes into the liquid!
    exc_inputs = range(50)
    inh_inputs = []
    exc_weights = [AINP*1E-9]*50
    inh_weights = []
    MaassTest.update(exc_inputs, inh_inputs, exc_weights, inh_weights)

    # Appends to the output
    tstp,spks = MaassTest.verify_spikes()
    t.append(tstp)
    spikes.append(spikes_idx[spks])
    
    for i in range(450):
        # Runs one step and save the result
        MaassTest.update([], [], [], [])

        # Appends to the output
        tstp,spks = MaassTest.verify_spikes()
        t.append(tstp)
        spikes.append(spikes_idx[spks])
    
    return t,spikes


# In[7]:

# %timeit t,spikes = compare_to_brian()
t,spikes = compare_to_brian()


# In[10]:

x_plot = numpy.array([t[ti] for i,ti in zip(spikes,xrange(len(spikes))) for j in i])
y_plot = [j for i,ti in zip(spikes,xrange(len(spikes))) for j in i]

plt.figure();
plt.plot(x_plot*1000,y_plot,'.');
plt.xlim(t[0]*1000,t[-1]*1100);
plt.show();


### Speed tests!

# In[17]:

# Simply updates the network state, without receiving external spikes
get_ipython().magic(u'timeit MaassTest.update([], [], [], [])')


# In[18]:

# Update the network state processing 100 inhibitory spikes and 100 excitatory spikes
exc_inputs = numpy.random.randint(0,135,size=100)
inh_inputs = numpy.random.randint(0,135,size=100)
exc_weights = numpy.random.rand(100)*100E-9
inh_weights = numpy.random.rand(100)*100E-9
get_ipython().magic(u'timeit MaassTest.update(exc_inputs, inh_inputs, exc_weights, inh_weights)')


# #Comparison with Brian 1.4.X#
# http://briansimulator.org/

# In[28]:

import brian

try:
    brian.reinit() # Necessary when using a python console
except:
    print "Brian REINIT error..."

import numpy # I could use "brian.", because Brian imports numpy, but I prefer not.
import time

import lsm_connections_probability as lm # Creates the 3D Grid and the connections according to Maass 2002

import lsm_dynamical_synapses_v1 as ls # Creates the dynamical synapses according to Maass 2002 and using the output 
                                       # from the lsm_connections_probability    


Net_shape=(15,3,3)
Number_of_neurons_lsm=135

my_STEP = 0.2
STP_OFF = True # Tells the system to use or not STP in the liquid
lbd_value =1.2 # lbd controls the connections probabilities

AINP = 100 # Input gains in nA

# These make easier to use the Brian objects without the "brian." at the beginning
ms = brian.ms
mV = brian.mV
nA = brian.nA
nF = brian.nF
NeuronGroup = brian.NeuronGroup
SpikeGeneratorGroup = brian.SpikeGeneratorGroup
Synapses = brian.Synapses
SpikeMonitor = brian.SpikeMonitor
network_operation = brian.network_operation 
defaultclock = brian.defaultclock

defaultclock.dt = my_STEP*ms



initial_time = time.time()

print "Initial time (in seconds):",initial_time

print "#"*78
print "#"*78
print "Liquid State Machine - 2 DoF arm experiments!"
print "#"*78
print "#"*78



lsm_3dGrid_flat = numpy.zeros(Number_of_neurons_lsm) 
# This creates a numpy 1D array with 'Number_of_neurons_lsm' positions
# I'm using a numpy array to be able to use the reshape method to 
# change from 1D (vector) to 3D (matrix)


#
# Number of Inhibitory neurons - LIQUID - 20% of the total neurons
lsm_indices = numpy.arange(Number_of_neurons_lsm) # Generates a list of indices

numpy.random.seed(my_seed) # Forces the numpy to seed the random generator

numpy.random.shuffle(lsm_indices) # Shuffles the list

inhibitory_index_L = numpy.array(lsm_indices[:int(Number_of_neurons_lsm*0.2)]) # The first (approx.) 20% as inhibitory
excitatory_index = numpy.array(lsm_indices[int(Number_of_neurons_lsm*0.2):])# The rest of the neurons as excitatory


# This is the dictionary that has all the connections parameters according to Maass 2002.
# It is necessary to create the 3D connections and the STP configuration matrices
# E=>1 (excitatory) and I=>0 (inhibitory)
# Ex.: (0,0) => II
# Dynamical Synapses Parameters (STP):
Connections_Parameters={
              (0,0):[ # II
                      0.1,       # CGupta=0.1        # Parameter used at the connection probability - from Maass2002 paper
                      0.32,      # UMarkram=0.32     # Use (U) - Parameter used at the Dynamic Synapse - from Maass2002 paper
                      0.144,     # DMarkram=0.144    # Time constant for Depression (tau_rec) - used at the Dynamic Synapse - from Maass2002 paper                    
                      0.06,      # FMarkram=0.06     # Time constant for Facilitation (tau_facil) - used at the Dynamic Synapse - from Maass2002 paper
                      2.8,       # AMaass=2.8        # (nA) In the Maass2002 paper the value is negative, but because I need a positive scale (random.normal parameter) and there is a negative sign in front of the abs function I changed this to positive
                      0.8        # Delay_trans = 0.8 # In Maass paper the transmission delay is 0.8 to II, IE and EI        
                  ],
              (0,1):[ # IE
                      0.4,    # CGupta=0.4
                      0.25,   # UMarkram=0.25
                      0.7,    # DMarkram=0.7
                      0.02,   # FMarkram=0.02
                      3.0,    # AMaass=3.0 #in the Maass2002 paper the value is negative, but because I need a positive scale (random.normal parameter) and there is a negative sign in front of the abs function I changed this to positive
                      0.8     # Delay_trans = 0.8 # in Maass paper the transmission delay is 0.8 to II, IE and EI
                  ],
              (1,0):[ # EI
                      0.2,    # CGupta=0.2
                      0.05,   # UMarkram=0.05
                      0.125,  # DMarkram=0.125
                      1.2,    # FMarkram=1.2
                      1.6,    # AMaass=1.6
                      0.8     # Delay_trans = 0.8 # in Maass paper the transmission delay is 0.8 to II, IE and EI
                  ],
              (1,1):[ # EE
                      0.3,    # CGupta=0.3 
                      0.5,    # UMarkram=0.5
                      1.1,    # DMarkram=1.1
                      0.05,   # FMarkram=0.05
                      1.2,    # AMaass=1.2 #scaling parameter or absolute synaptic efficacy or weight - from Maass2002 paper
                      1.5     # Delay_trans = 1.5 # in Maass paper the transmission delay is 1.5 to EE connection
                  ]
              }


# Utilizes the functions in the lsm_connections_probability.py
# =>output = {'exc':connections_list_exc,'inh':connections_list_inh, '3Dplot_a':positions_list_a, '3Dplot_b':positions_list_b}
# connections_list_exc= OR connections_list_inh=
  # ((i,j), # PRE and POS synaptic neuron indexes
  # pconnection, # probability of the connection
  # (W_n, U_ds, D_ds, F_ds), # parameters according to Maass2002
  # Delay_trans, # parameters according to Maass2002
  # connection_type)

# Generate the connections matrix inside the Liquid (Liquid->Liquid) - according to Maass2002
#
print "Liquid->Liquid connections..."

numpy.random.seed(my_seed) # Forces the numpy to seed the random generator

print "Generating the Liquid->Liquid connections..."
output_L_L = lm.generate_connections(lsm_3dGrid_flat, inhibitory_index_L, Net_shape, 
                                  CParameters=Connections_Parameters, lbd=lbd_value) # lbd controls the connections

print "Liquid->Liquid connections...Done!"

#
# These are the cell (neuron) parameters according to Maass 2002
#
cell_params_lsm = {  'cm'        : 30*nF,    # Capacitance of the membrane 
                                           # =>>>> MAASS PAPER DOESN'T MENTION THIS PARAMETER DIRECTLY
                                            #       but the paper mention a INPUT RESISTANCE OF 1MEGA Ohms and tau_m=RC=30ms, so cm=30nF
                   'i_offset'  : 0.0*nA,   # Offset current - random for each neuron from [14.975nA to 15.025nA] => Masss2002 - see code below
                   'tau_m'     : 30.0*ms,  # Membrane time constant => Maass2002
                   'tau_refrac_E': 3.0*ms, # Duration of refractory period - 3mS for EXCITATORY => Maass2002
                   'tau_refrac_I': 2.0*ms, # Duration of refractory period - 2mS for INHIBITORY => Maass2002
                   'tau_syn_E' : 3.0*ms,   # Decay time of excitatory synaptic current => Maass2002
                   'tau_syn_I' : 6.0*ms,   # Decay time of inhibitory synaptic current => Maass 2002
                   'v_reset'   : 13.5*mV,  # Reset potential after a spike => Maass 2002
                   'v_rest'    : 0.0*mV,   # Resting membrane potential => Maass 2002
                   'v_thresh'  : 15.0*mV,  # Spike threshold => Maass 2002
                   'i_noise'   : 1.0*nA    # Used in Joshi 2005: mean 0 and SD=1nA
                }

# IF_curr_exp - MODEL EXPLAINED
# Leaky integrate and fire model with fixed threshold and
# decaying-exponential post-synaptic current. 
# (Separate synaptic currents for excitatory and inhibitory synapses)
lsm_neuron_eqs='''
  dv/dt  = (ie + ii + i_offset + i_noise)/c_m + (v_rest-v)/tau_m : mV
  die/dt = -ie/tau_syn_E                : nA
  dii/dt = -ii/tau_syn_I                : nA
  tau_syn_E                             : ms
  tau_syn_I                             : ms
  tau_m                                 : ms
  c_m                                   : nF
  v_rest                                : mV
  i_offset                              : nA
  i_noise                               : nA
  '''
# lsm_neuron_eqs='''
#   dv/dt  = (ie + ii + i_offset + i_noise)/c_m + (v_rest-v)/tau_m : mV
#   ie                                    : nA
#   ii                                    : nA
#   tau_m                                 : ms
#   c_m                                   : nF
#   v_rest                                : mV
#   i_offset                              : nA
#   i_noise                               : nA
#   '''



########################################################################################################################
#
# LIQUID - Setup
#
print "LIQUID - Setup..."

# Creates a vector with the corresponding refractory period according to the type of neuron (inhibitory or excitatory)
# IT MUST BE A NUMPY ARRAY OR BRIAN GIVES CRAZY ERRORS!!!!!
refractory_vector = [ cell_params_lsm['tau_refrac_E'] ]*Number_of_neurons_lsm # fills the list with the value corresponding to excitatory neurons
for i in range(Number_of_neurons_lsm):
  if i in inhibitory_index_L:
      refractory_vector[i]=cell_params_lsm['tau_refrac_I'] # only if the neuron is inibitory, changes the refractory period value!

refractory_vector=numpy.array(refractory_vector) # Here it is converted to a NUMPY ARRAY


# This is the population (neurons) used exclusively to the Liquid (pop_lsm).
# All the neurons receive the same threshold and reset voltages.
pop_lsm = NeuronGroup(Number_of_neurons_lsm, model=lsm_neuron_eqs, 
                                             threshold=cell_params_lsm['v_thresh'], 
                                             reset=cell_params_lsm['v_reset'], 
                                             refractory=refractory_vector, 
                                             max_refractory=max(cell_params_lsm['tau_refrac_E'], 
                                                                cell_params_lsm['tau_refrac_I']))


# Here I'm mixing numpy.fill with the access of the state variable "c_m" in Brian (because Brian is using a numpy.array)
# Sets the value of the capacitance according to the cell_params_lsm (same value to all the neurons)
pop_lsm.c_m.fill(cell_params_lsm['cm'])


# Sets the value of the time constant RC (or membrane constant) according to the cell_params_lsm (same value to all the neurons)
pop_lsm.tau_m.fill(cell_params_lsm['tau_m'])

# Sets the i_offset according to Maass2002
# The i_offset current is random, but never changes during the simulation.
# this current should be drawn from a uniform distr [14.975,15.025]
# Joshi2005 does [13.5,14.5] ???? Maybe is to avoid spikes without inputs...
numpy.random.seed(my_seed) # Forces the numpy to seed the random generator
pop_lsm.i_offset=numpy.random.uniform(14.975,15.025, Number_of_neurons_lsm)*nA



pop_lsm.tau_syn_E.fill(cell_params_lsm['tau_syn_E']) # (same value to all the neurons)
pop_lsm.tau_syn_I.fill(cell_params_lsm['tau_syn_I']) # (same value to all the neurons)


# All neurons receive the same value to the resting voltage.
pop_lsm.v_rest.fill(cell_params_lsm['v_rest']) # (same value to all the neurons)


# Sets the initial membrane voltage according to Maass2002. Doesn't change during the simulation.
# this current should be drawn from a uniform distr [13.5mV,15.0mV]
numpy.random.seed(my_seed) # Forces the numpy to seed the random generator
pop_lsm.v=numpy.random.uniform(13.5,15.0, Number_of_neurons_lsm)*mV


#
# Loading or creating the Synapses objects used within the Liquid
print "Liquid->Liquid connections..."

syn_lsm_obj = ls.LsmConnections(pop_lsm, pop_lsm, output_L_L, nostp=STP_OFF)

# Generates the Liquid->Liquid - EXCITATORY synapses
syn_lsm_exc = syn_lsm_obj.create_synapses('exc')

# Generates the Liquid->Liquid - INHIBITORY synapses
syn_lsm_inh = syn_lsm_obj.create_synapses('inh')


print "Liquid->Liquid connections...Done!"

total_number_of_connections_liquid = len(syn_lsm_exc) + len(syn_lsm_inh)

print "Number of inhibitory synapses in the Liquid: " + str(len(syn_lsm_inh)) # DEBUG to verify if it is working
print "Number of excitatory synapses in the Liquid: " + str(len(syn_lsm_exc)) # DEBUG to verify if it is working


# To understand what is being returned:
# pop_lsm: it is necessary to connect the neuron network with the rest of the world
# [syn_lsm_obj, syn_lsm_exc, syn_lsm_inh]: to include these objects at the simulation (net=Net(...); net.run(total_sim_time*ms)); 
# It is a list because is easy to concatenate lists :D

print "LIQUID - Setup...Done!"

#
# End of the LIQUID - Setup
########################################################################################################################



########################################################################################################################
#
# INPUT - Setup
#
print "INPUT - Setup..."

tspk = defaultclock.dt*50 # The neurons spike after 50 time steps!
number_of_spikes = 50

spiketimes = [(i,tspk) for i in range(number_of_spikes)] 
                # The spikes are going to be received during the simulation, 
                # so this is always an empty list when using the step_by_step_brian_sim!


# I'm using only one big input layer because Brian docs say it is better for the performance
SpikeInputs = SpikeGeneratorGroup(number_of_spikes, spiketimes)


#
#
# Here the synapses are created. The synapses created are ALWAYS excitatory because it is 
# connecting through 'ie' in the neuron model!

syn_world_Input = Synapses(SpikeInputs, pop_lsm,
                                     model='''w : 1''',
                                     pre='''ie+=w''')


for i in range(len(spiketimes)):
    syn_world_Input[i,i] = True

weights_input_liquid = numpy.array([AINP]*number_of_spikes)*nA

syn_world_Input.w = weights_input_liquid
syn_world_Input.delay=0*ms

print "INPUT - Setup...Done!"

#
# End of the INPUT - Setup (creation of the connections between the Poisson input and the Liquid!)
#
########################################################################################################################


# DON'T FORGET TO INSERT THE FUNCTION "generate_i_noise" INTO THE MONITORS_OBJECT LIST!!!!
# Generates the noisy current at each time step (as seen in Joshi2005)
@network_operation(clock=defaultclock)
def generate_i_noise():
    # These are the noise currents inside each liquid's neuron
    pop_lsm.i_noise=numpy.random.normal(loc=0, scale=cell_params_lsm['i_noise'],size=Number_of_neurons_lsm)*nA

populations_sim = [pop_lsm, SpikeInputs]

synapses_sim = [syn_lsm_exc, syn_lsm_inh, syn_world_Input]

Input_layer, Output_layer, pop_objects, syn_objects, monitors_objects = SpikeInputs, pop_lsm, populations_sim, synapses_sim, [generate_i_noise]

OutputMonitor=brian.SpikeMonitor(pop_lsm, record=True)
VMonitor=brian.StateMonitor(pop_lsm,'v', record=True)
IEMonitor=brian.StateMonitor(pop_lsm,'ie', record=True)

net = brian.Network(pop_objects + syn_objects + monitors_objects + [OutputMonitor,VMonitor,IEMonitor])

print "Setup time:", time.time()-initial_time
initial_time = time.time()

net.run(500*my_STEP*ms)
print "Simulation time:", time.time()-initial_time


# In[29]:

plt.figure()
brian.raster_plot(OutputMonitor)
plt.xlim(0,100)
plt.show()


# In[30]:

fig=plt.figure()
plt.subplot(211)
brian.raster_plot(OutputMonitor,markersize=6)
plt.title("Brian")
plt.xlim(0,100)
plt.xlabel("")
plt.subplot(212)
plt.plot(x_plot*1000,y_plot,'.',markersize=4.5)
plt.title("Simulator")
plt.xlabel("time(ms)")
plt.xlim(0,100)
plt.show();
fig.subplots_adjust(hspace=4) # Adjust the distance between subplots


# In[ ]:




# -*- coding: utf-8 -*-

import brian
import numpy

# This code was written for Brian 1.4.x
# I've never tried it with the Brian 2.x

class LsmConnections(object):

    """
    Generate the synapses (connections) according to Maass 2002 paper
    LsmConnections(brian_neuron_group, output_from_lsm_module)

    create_synapses(synapse_type)
    synapse_type => 'exc' or 'inh'
    'exc' => EE and EI
    'inh' => II and IE

    """

    def __init__(self, pop_lsm_a, pop_lsm_b, output, **xtra):
        self.pop_lsm_a = pop_lsm_a
        self.pop_lsm_b = pop_lsm_b
        self.output = output
        self.sim_clock = brian.defaultclock
        self.nostp = False

        if xtra:
            for k in xtra:
                if k=="clock":
                    self.sim_clock = xtra["clock"]
                    print "Using supplied clock..."

                if k=="nostp":
                    self.nostp=xtra["nostp"]


    def create_synapses(self, synapse_type):
        #
        # Creates the connections (Synapse object) among the neurons in the liquid
        #

        self.syn_type = synapse_type

        # synapse_type = 'exc'
        # EXCITATORY (PRE) TO ANYTHING (POST) => ie+=w*u*x (what defines if it is a Excitatory or Inhibitory connection 
        # is the place where it is connected - ie or ii)
        # EE and IE connection types

        # synapse_type = 'inh'
        # INHIBITORY (PRE) TO ANYTHING (POST) => ii+=w*u*x (what defines if it is a Excitatory or Inhibitory connection 
        # is the place where it is connected - ie or ii)
        # IE and II connection types        

        # STP equation from: 
        # http://www.briansimulator.org/docs/synapses.html
        # http://www.briansimulator.org/docs/examples-synapses_short_term_plasticity.html
        # In this part is selected the right synapse equation according to the type of the connection (see comments above)
        # The ONLY difference between the EXCITATORY AND THE INHIBITORY is where the weight is injected: ie or ii
        if self.syn_type=='exc' and not self.nostp:
                model_eq='''x : 1
                            u : 1
                            w : 1
                            tauf : 1
                            taud : 1
                            U : 1
                            '''
                pre_eq='''u=U+(u-U)*numpy.exp(-(t-lastupdate)/tauf)
                          x=1+(x-1)*numpy.exp(-(t-lastupdate)/taud)
                          ie+=w*u*x
                          x*=(1-u)
                          u+=U*(1-u)'''

        elif self.syn_type=='exc' and self.nostp:
                model_eq='''w : 1'''
                pre_eq='''ie+=w'''
                # pre_eq='''v+=w/c_m'''

        elif self.syn_type=='inh' and not self.nostp:
                model_eq='''x : 1
                            u : 1
                            w : 1
                            tauf : 1
                            taud : 1
                            U : 1
                            '''            
                pre_eq='''u=U+(u-U)*numpy.exp(-(t-lastupdate)/tauf)
                          x=1+(x-1)*numpy.exp(-(t-lastupdate)/taud)
                          ii+=w*u*x
                          x*=(1-u)
                          u+=U*(1-u)'''

        elif self.syn_type=='inh' and self.nostp:
                model_eq='''w : 1'''
                pre_eq='''ii+=w'''
                # pre_eq='''v+=w/c_m'''
        
        self.syn_lsm=brian.Synapses(self.pop_lsm_a,self.pop_lsm_b,
                   model=model_eq,
                   pre=pre_eq,
                   clock=self.sim_clock)


        # Sets the synapses according to the probability function (Maass 2002 - lsm_module.py)
        # NOT OPTMIZED!!!
        # Caution about creation of synapses:
        # 1) there is no deletion
        # 2) synapses are added, not replaced (e.g. S[1,2]=True;S[1,2]=True creates 2 synapses)

        self.w_syn=[] # list to store the Weights (w) in the same order as the creation of the synapses
        self.U_syn=[] # list to store the Use (U) in the same order as the creation of the synapses
        self.taud_syn=[] # list to store the Time constant for Depression (taud) in the same order as the creation of the synapses
        self.tauf_syn=[] # list to store the Time constant for Facilitation (tauf) in the same order as the creation of the synapses
        self.d_syn=[] # list to store the Delay (D) in the same order as the creation of the synapses


        # This part goes through all the connections (EE, EI, II and IE) but sets only the ones that start according to 'synapse_type'
        for i in xrange(len(self.output[self.syn_type])):
            
            ipre,ipos = self.output[self.syn_type][i][0] # sets the position of the neurons
            self.syn_lsm[ipre,ipos] = True # here is where the synapse is really created

            # It is extremely important to append this values at the same order the synapses (connections) are created otherwise they will not match the right synapse!!!!!!
            self.w_syn.append(self.output[self.syn_type][i][2][0]) # sets the value of the Weight (w) for this connection
            self.U_syn.append(self.output[self.syn_type][i][2][1]) # sets the value of the Use (U) for this connection
            self.taud_syn.append(self.output[self.syn_type][i][2][2]) # sets the value of the Time constant for Depression (taud) for this connection
            self.tauf_syn.append(self.output[self.syn_type][i][2][3]) # sets the value of the Time constant for Facilitation (tauf) for this connection
            self.d_syn.append(self.output[self.syn_type][i][3]) # sets the value of the Delay (D) for this connection

        # The output[self.syn_type] is organized like this:
        #     (i,j), # PRE and POS synaptic neuron indexes (this is not the same thing as the position in the 3D Grid)
        #     pconnection, # probability of the connection (according to the Maass 2002 equation)
        #     (W_n, U_ds, D_ds, F_ds), # parameters according to Maass2002
        #     Delay_trans, # parameters according to Maass2002
        #     connection_type

        # What is the order of these vectors below in relation to the synapses connections????
        # THESE VECTORS SEEM TO FOLLOW THE SAME ORDER AS THE CREATION OF THE SYNAPSES
        # BUT I COULDN'T FIND ANY OFFICIAL INFORMATION ABOUT THAT
        # => NEED TO BE CHECKED BETTER IN THE BRIAN SIMULATOR SOURCE CODE!!! 
        self.syn_lsm.taud=numpy.array(self.taud_syn)*brian.ms
        self.syn_lsm.tauf=numpy.array(self.tauf_syn)*brian.ms
        self.syn_lsm.U=numpy.array(self.U_syn)
        self.syn_lsm.w=numpy.array(self.w_syn)*brian.nA
        self.syn_lsm.delay=numpy.array(self.d_syn)*brian.ms

        self.syn_lsm.u=numpy.array(self.U_syn) # Considering u<=U at the initialization

        self.syn_lsm.x=1 # In Joshi thesis he uses u1=U and x1=1 (x1 is the R1 in his thesis)

        # from: http://www.scholarpedia.org/article/Short-term_synaptic_plasticity
        # In the model proposed by Tsodyks and Markram (Tsodyks 98), the STD effect is 
        # modeled by a normalized variable x (0≤x≤1), denoting the fraction of resources 
        # that remain available after neurotransmitter depletion. The STF effect is modeled 
        # by a utilization parameter u, representing the fraction of available resources ready 
        # for use (release probability). Following a spike, (i) u increases due to spike-induced 
        # calcium influx to the presynaptic terminal, after which (ii) a fraction u of available 
        # resources is consumed to produce the post-synaptic current. Between spikes, u decays 
        # back to zero with time constant τf and x recovers to 1 with time constant τd.
        #
        # In general, an STD-dominated synapse favors information transfer for low firing rates, 
        # since high-frequency spikes rapidly deactivate the synapse
        # 
        # Since STP has a much longer time scale than that of single neuron dynamics (the latter is typically 
        # in the time order of 10−20 milliseconds), a new feature STP can bring to the network dynamics is 
        # prolongation of neural responses to a transient input.
        # 
        # The interplay between the dynamics of u and x determines whether the joint effect of ux is dominated by 
        # depression or facilitation. In the parameter regime of τd≫τf and large U, an initial spike incurs a large 
        # drop in x that takes a long time to recover; therefore the synapse is STD-dominated (Fig.1B). 
        # In the regime of τf≫τd and small U, the synaptic efficacy is increased gradually by spikes, and consequently 
        # the synapse is STF-dominated (Fig.1C). This phenomenological model successfully reproduces the kinetic 
        # dynamics of depressed and facilitated synapses observed in many cortical areas.

        return self.syn_lsm
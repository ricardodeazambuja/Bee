import numpy

# This code is NOT optimized!
# I wrote it thinking about the readability...


def euclidean_distance(pointA, pointB):
	"""	
	Calculates the Euclidean Distance between two 3D points
	pointA = tuple(x1,y1,z1)
	pointB = tuple(x2,y2,z2)
	Needs numpy as numpy!
	"""
	x1,y1,z1 = pointA
	x2,y2,z2 = pointB

	return numpy.sqrt(numpy.power((x1-x2),2)+numpy.power((y1-y2),2)+numpy.power((z1-z2),2))

def calculate_connection_probability():
	pass


def generate_connections(flattened_3DGrid_a, inhibitory_index_vector_a, Net_shape_a, flattened_3DGrid_b=None, inhibitory_index_vector_b=None, Net_shape_b=None, CParameters=None, lbd=1.2, randomstate=numpy.random.RandomState()):
	"""
	Generates the necessary data to create an 1D connected neuron network with the probabilities of the 3D connections according to Maass2002 paper.
	Normally this could be done using the pyNN.space.3DGrid, but SpiNNaker doesn't accept this. That's why it is
	necessary to flatten everything to a 1D vector in the end. Because the others simulators are not going to complain
	about the 1D connection I think it is better to keep compatibility to SpiNNaker.
	Also all the parameters with dependence to the connection type are setup in this function (according to Maass2002)

	flattened_3DGrid = numpy array
	inhibitory_index_vector = 1D vector with the index from flattened_3DGrid indicating the neuron is inhibitory
	Net_shape = tuple with the 3D shape of the neuron network (x,y,z)
	lbd = is the lambda factor from Maass paper. Controls both the average number of connections and the average distance between neurons that are synaptically connected
	CParameters = dictionary with the STP parameters to each type of connection ('II', 'IE', 'EE' and 'EI').

	It is also possible to generate connections between two groups: a and b. In this situation both groups are positioned in 
	such a way that they don't overlap and are centralized, but the system ONLY generates the connections between the two groups (no internal connections).
	This option is useful if someone wants to connect the liquid to another SNN using the same probability calculations to generate the connections.

	It returns a dictionary:
	{'exc':connections_list_exc,'inh':connections_list_inh, '3Dplot_a':positions_list_a, '3Dplot_b':positions_list_b} where
	connections_list_*:
						(
						(i,j), # PRE and POS synaptic neuron indexes (according to the input flattened_3DGrid)
						pconnection, # probability of the connection calculated
						(W_n, U_ds, D_ds, F_ds), # parameters according to Maass2002
						Delay_trans, # parameters according to Maass2002
						connection_type # tuple (A,B) where 1=>excitatory and 0=>inhibitory
						)

	positions_list_*:
	list with x,y,z tuples of the corresponding position to be used with the Euclidean Distance

	"""

	#Transforms the 1D vector in a 3D according to Net_shape using the reshape method from numpy array
	#The reshape method takes the components of the vector and fills the matrix in a sequential way.
	#Ex: [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] 
	#    Shape => (3,4,2) 
	#    Result => 
	# array([[[ 0,  1],
	#         [ 2,  3],
	#         [ 4,  5],
	#         [ 6,  7]],
	#        [[ 8,  9],
	#         [10, 11],
	#         [12, 13],
	#         [14, 15]],
	#        [[16, 17],
	#         [18, 19],
	#         [20, 21],
	#         [22, 23]]])


	
	if not (flattened_3DGrid_b is None): 
		pop_b_y_offset = Net_shape_a[1] # Translates the second population for the Y lenght of the first when they are not the same
		                                # So they are not going to overlap in the calculation of the distance or when plotting
		pop_a_y_offset = 0 # It is not necessary to translate the first population in this axis!

	else: 
	    # This is for the case where the connections ARE in the same population
		# Copy the variables from population 'a' to population 'b'
		flattened_3DGrid_b = flattened_3DGrid_a
		inhibitory_index_vector_b = inhibitory_index_vector_a
		pop_b_y_offset=0 # In this case it is not necessary a offset in Y because 'a' and 'b' are the same population!
		pop_a_y_offset=0
		Net_shape_b = Net_shape_a


	new_3dGrid_a = flattened_3DGrid_a.reshape(Net_shape_a) # Converts to a 3D shape (matrix) the 1D list of population 'a'
	new_3dGrid_b = flattened_3DGrid_b.reshape(Net_shape_b) # Converts to a 3D shape (matrix) the 1D list of population 'b'

	pop_a_x_offset = -Net_shape_a[0]/2.0   # Centralize in the x=0 axis
	pop_a_z_offset = -Net_shape_a[2]/2.0   # Centralize in the z=0 axis
	
	pop_b_x_offset = -Net_shape_b[0]/2.0   # Centralize in the x=0 axis
	pop_b_z_offset = -Net_shape_b[2]/2.0   # Centralize in the z=0 axis



	# Generates a list with x,y,z tuples of the corresponding position to be used with the Euclidean Distance
	positions_list_a = [(i+pop_a_x_offset, j+pop_a_y_offset, k+pop_a_z_offset) for i in range(new_3dGrid_a.shape[0]) for j in range(new_3dGrid_a.shape[1]) for k in range(new_3dGrid_a.shape[2])]
	positions_list_b = [(i+pop_b_x_offset, j+pop_b_y_offset, k+pop_b_z_offset) for i in range(new_3dGrid_b.shape[0]) for j in range(new_3dGrid_b.shape[1]) for k in range(new_3dGrid_b.shape[2])]


	connections_list_exc=[] # creates the list to be used to append the connection data of the excitatory neurons
	connections_list_inh=[] # creates the list to be used to append the connection data of the inhibitory neurons

	# Signalize the type of the pre and post synaptics neurons at each connection
	# 1 => excitatory
	# 0 => inhibitory
	connection_type=(None,None)

	# Go through all the neurons to apply the probability equation according to Maass2002 paper( C*exp(-D^2(a,b)/lambda^2) )
	# Also calculates the values of U, D and F (used with the dynamic synapse) and the A (the weight) according Maass2002
	# Here is explicit that there is an order from the population 'a' to the population 'b' and it is one to all connection. 
	# In the case of the liquid, both populations are the same.
	for i in range(len(flattened_3DGrid_a)):
		for j in range(len(flattened_3DGrid_b)):
			#Test to choose the right constant (from Maass paper <= Gupta paper)
			if i in inhibitory_index_vector_a:
				#INHIBITORY CONNECTIONS
				#II or IE
				if j in inhibitory_index_vector_b:
					#II
					(Tpre,Tpos)=(0,0)
					CGupta=CParameters[(Tpre,Tpos)][0] 			# Parameter used at the connection probability - from Maass2002 paper
					UMarkram=CParameters[(Tpre,Tpos)][1]     	# Use (U) - Parameter used at the Dynamic Synapse - from Maass2002 paper
					DMarkram=CParameters[(Tpre,Tpos)][2]    	# (second) Time constant for Depression (tau_rec) - used at the Dynamic Synapse - from Maass2002 paper					
					FMarkram=CParameters[(Tpre,Tpos)][3]    	# (second) Time constant for Facilitation (tau_facil) - used at the Dynamic Synapse - from Maass2002 paper
					AMaass=CParameters[(Tpre,Tpos)][4]       	# (nA) In the Maass2002 paper the value is negative, but because I need a positive scale (random.normal parameter) and there is a negative sign in front of the abs function I changed this to positive
					Delay_trans=CParameters[(Tpre,Tpos)][5] 	# (msecond) In Maass paper the transmission delay is 0.8 to II, IE and EI and 1.5 to EE


					U_ds=abs(randomstate.normal(loc=UMarkram, scale=UMarkram/2)) # Because a gaussian goes from -inf to +inf I need to keep it positive
					D_ds=abs(randomstate.normal(loc=DMarkram, scale=DMarkram/2))
					F_ds=abs(randomstate.normal(loc=FMarkram, scale=FMarkram/2))
					W_n=-abs(randomstate.normal(loc=AMaass, scale=AMaass/2)) # Because AMaass is negative (inhibitory) so is inserted the "-" here

					connection_type=(Tpre,Tpos)

				else:
					#IE
					(Tpre,Tpos)=(0,1)
					CGupta=CParameters[(Tpre,Tpos)][0] 			# Parameter used at the connection probability - from Maass2002 paper
					UMarkram=CParameters[(Tpre,Tpos)][1]     	# Use (U) - Parameter used at the Dynamic Synapse - from Maass2002 paper
					DMarkram=CParameters[(Tpre,Tpos)][2]    	# Time constant for Depression (tau_rec) - used at the Dynamic Synapse - from Maass2002 paper					
					FMarkram=CParameters[(Tpre,Tpos)][3]    	# Time constant for Facilitation (tau_facil) - used at the Dynamic Synapse - from Maass2002 paper
					AMaass=CParameters[(Tpre,Tpos)][4]       	# (nA) In the Maass2002 paper the value is negative, but because I need a positive scale (random.normal parameter) and there is a negative sign in front of the abs function I changed this to positive
					Delay_trans=CParameters[(Tpre,Tpos)][5] 	# In Maass paper the transmission delay is 0.8 to II, IE and EI and 1.5 to EE


					U_ds=abs(randomstate.normal(loc=UMarkram, scale=UMarkram/2))
					D_ds=abs(randomstate.normal(loc=DMarkram, scale=DMarkram/2))
					F_ds=abs(randomstate.normal(loc=FMarkram, scale=FMarkram/2))
					W_n=-abs(randomstate.normal(loc=AMaass, scale=AMaass/2)) # Because AMaass is negative (inhibitory) so is inserted the "-" here
					
					connection_type=(Tpre,Tpos)

			#EXCITATORY CONNECTIONS
			elif j in inhibitory_index_vector_a:
				#EI
				(Tpre,Tpos)=(1,0)
				CGupta=CParameters[(Tpre,Tpos)][0] 			# Parameter used at the connection probability - from Maass2002 paper
				UMarkram=CParameters[(Tpre,Tpos)][1]     	# Use (U) - Parameter used at the Dynamic Synapse - from Maass2002 paper
				DMarkram=CParameters[(Tpre,Tpos)][2]    	# Time constant for Depression (tau_rec) - used at the Dynamic Synapse - from Maass2002 paper					
				FMarkram=CParameters[(Tpre,Tpos)][3]    	# Time constant for Facilitation (tau_facil) - used at the Dynamic Synapse - from Maass2002 paper
				AMaass=CParameters[(Tpre,Tpos)][4]       	# (nA) In the Maass2002 paper the value is negative, but because I need a positive scale (random.normal parameter) and there is a negative sign in front of the abs function I changed this to positive
				Delay_trans=CParameters[(Tpre,Tpos)][5] 	# In Maass paper the transmission delay is 0.8 to II, IE and EI and 1.5 to EE


				U_ds=abs(randomstate.normal(loc=UMarkram, scale=UMarkram/2))
				D_ds=abs(randomstate.normal(loc=DMarkram, scale=DMarkram/2))
				F_ds=abs(randomstate.normal(loc=FMarkram, scale=FMarkram/2))
				W_n=abs(randomstate.normal(loc=AMaass, scale=AMaass/2))			
				
				connection_type=(Tpre,Tpos)

			else:
				# EE
				(Tpre,Tpos)=(1,1)
				CGupta=CParameters[(Tpre,Tpos)][0] 			# Parameter used at the connection probability - from Maass2002 paper
				UMarkram=CParameters[(Tpre,Tpos)][1]     	# Use (U) - Parameter used at the Dynamic Synapse - from Maass2002 paper
				DMarkram=CParameters[(Tpre,Tpos)][2]    	# Time constant for Depression (tau_rec) - used at the Dynamic Synapse - from Maass2002 paper					
				FMarkram=CParameters[(Tpre,Tpos)][3]    	# Time constant for Facilitation (tau_facil) - used at the Dynamic Synapse - from Maass2002 paper
				AMaass=CParameters[(Tpre,Tpos)][4]       	# (nA) In the Maass2002 paper the value is negative, but because I need a positive scale (random.normal parameter) and there is a negative sign in front of the abs function I changed this to positive
				Delay_trans=CParameters[(Tpre,Tpos)][5] 	# In Maass paper the transmission delay is 0.8 to II, IE and EI and 1.5 to EE


				U_ds=abs(randomstate.normal(loc=UMarkram, scale=UMarkram/2))
				D_ds=abs(randomstate.normal(loc=DMarkram, scale=DMarkram/2))
				F_ds=abs(randomstate.normal(loc=FMarkram, scale=FMarkram/2))
				W_n=abs(randomstate.normal(loc=AMaass, scale=AMaass/2))

				connection_type=(Tpre,Tpos)

			# Calculate the probability of a connection to occur according to Maass2002 paper
			pconnection=CGupta*numpy.exp(
						-numpy.power(
							euclidean_distance(
								positions_list_a[i],
								positions_list_b[j]
								)
							,2) / numpy.power(lbd,2)
						)

			# According to the probability "pconnection", sets "connected" to zero or one
			if randomstate.random_sample() <= pconnection:
				connected=1
			else:
				connected=0


			# Generates the list with the information about the connections
			if connected:
				if connection_type[0]==0: #if IX (inhibitory connection)
					connections_list_inh.append(
						(
						(i,j), # PRE and POS synaptic neuron indexes
						pconnection, # probability of the connection
						(W_n, U_ds, D_ds, F_ds), # parameters according to Maass2002
						Delay_trans, # parameters according to Maass2002
						connection_type
						)
					)
				else: #if EX (excitatory connection)
					connections_list_exc.append(
						(
						(i,j),
						pconnection,
						(W_n, U_ds, D_ds, F_ds), 
						Delay_trans,
						connection_type
						)
					)

	return {'exc':connections_list_exc,'inh':connections_list_inh, '3Dplot_a':positions_list_a, '3Dplot_b':positions_list_b}
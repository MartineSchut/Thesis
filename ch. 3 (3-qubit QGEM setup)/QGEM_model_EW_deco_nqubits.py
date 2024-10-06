import numpy as np
import matplotlib.pyplot as plt
from scipy import arange, conj, prod
#from qutip.qobj import Qobj
#from qutip.states import (state_index_number, state_number_index,
#                          state_number_enumerate)


def Ejk(j,k, dimension):
    matrix = np.zeros((dimension, dimension))
    matrix[j][k] = 1
    return matrix


def generalised_paulis_list(dimension):
    # source: http://mathworld.wolfram.com/GeneralizedGell-MannMatrix.html
    gen_paulis_list = []
    gen_paulis_list.append(np.identity(dimension))
    j = 0
    k = 0
    factor = np.sqrt(dimension/2.) #adapted from PhysRevA.83.032318
    for k in range(dimension):
        for j in range(dimension):
             if k<j:
                 gen_paulis_list.append(factor*(Ejk(j,k, dimension) + Ejk(k,j, dimension)))
             if k>j:
                 gen_paulis_list.append(-1.j*factor*(Ejk(j,k, dimension) - Ejk(k,j, dimension)))  
    for l in range(1, dimension):
        blank = np.zeros((dimension, dimension))
        for j in range(1, l + 1):
            blank = blank + Ejk(j-1,j-1, dimension) 
        blank = blank - (l)*Ejk(l,l, dimension)
        gen_paulis_list.append(factor*np.sqrt((2/(l*(l+1))))*blank)
    return gen_paulis_list

def gen_paulis_tensors(dimension, num_qudits):
    paulis_d = generalised_paulis_list(dimension)
    if num_qudits == 1:
        return paulis_d
    else:
        new_level = []
        next_level = gen_paulis_tensors(dimension, num_qudits - 1)
        for i in range(len(paulis_d)):
            for j in range((dimension**2)**(num_qudits - 1)):  
                new_level.append(np.kron(paulis_d[i], next_level[j]))
        return new_level

def all_letters(num_qubits):
    all_letters = ['I', 'X', 'Y', 'Z']
    one_letter = ['I', 'X', 'Y', 'Z']
    if num_qubits ==1:
        return one_letter
    else:
        for i in range(1, num_qubits):
            new_letters = []
            for letter in all_letters:
                for basis_elem in one_letter:
                    new_letters.append(letter + basis_elem)
            all_letters = new_letters
    return all_letters

def p_matrix_commute(p_mat_1, p_mat_2):
    if p_mat_1 == 'I':
        return 0
    elif p_mat_2 == 'I':
        return 0
    elif p_mat_1 == p_mat_2:
        return 0
    else: 
        return 1

def tpb_group(paulis):
    paulis_list = paulis[1:]
    color_dict = {'1': [paulis_list[0]]}
    color_id = 1
    for pauli in paulis_list[1:]:
       to_allocate = True
       for color in range(1, color_id + 1):
           count = 0
           for group_elem in color_dict[str(color)]:
               for id_pauli in range(len(paulis_list[0])):
                   count += p_matrix_commute(pauli[id_pauli], group_elem[id_pauli])
           if count == 0:
               color_dict[str(color)].append(pauli)
               to_allocate = False
               break
       if to_allocate:
           color_dict[str(color_id + 1)] = [pauli]
           color_id += 1
    return color_dict

def state_number_enumerate(dims, excitations=None, state=None, idx=0):
    """
    An iterator that enumerate all the state number arrays (quantum numbers on
    the form [n1, n2, n3, ...]) for a system with dimensions given by dims.

    Example:

        >>> for state in state_number_enumerate([2,2]):
        >>>     print(state)
        [ 0  0 ]
        [ 0  1 ]
        [ 1  0 ]
        [ 1  1 ]

    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.

    state : list
        Current state in the iteration. Used internally.

    excitations : integer (None)
        Restrict state space to states with excitation numbers below or
        equal to this value.

    idx : integer
        Current index in the iteration. Used internally.

    Returns
    -------
    state_number : list
        Successive state number arrays that can be used in loops and other
        iterations, using standard state enumeration *by definition*.

    """

    if state is None:
        state = np.zeros(len(dims), dtype=int)

    if excitations and sum(state[0:idx]) > excitations:
        pass
    elif idx == len(dims):
        if excitations is None:
            yield np.array(state)
        else:
            yield tuple(state)
    else:
        for n in range(dims[idx]):
            state[idx] = n
            for s in state_number_enumerate(dims, excitations, state, idx + 1):
                yield s


def state_number_index(dims, state):
    """
    Return the index of a quantum state corresponding to state,
    given a system with dimensions given by dims.

    Example:

        >>> state_number_index([2, 2, 2], [1, 1, 0])
        6

    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.

    state : list
        State number array.

    Returns
    -------
    idx : int
        The index of the state given by `state` in standard enumeration
        ordering.

    """
    return int(
        sum([state[i] * prod(dims[i + 1:]) for i, d in enumerate(dims)]))


class Physics:
    def __init__(self, gravity, h_bar):
        self.gravity = gravity
        self.h_bar = h_bar

class Model_Lattice:
    def __init__(self, lattice_x, lattice_y):
        self.lattice_x = lattice_x
        self.lattice_y = lattice_y
        self.qudit_list = []

    def add_qudit(self, qudit):
        #TODO assert that the qudit fits in the lattice
        #TODO model constraints
        self.qudit_list.append(qudit)

    def plot_lattice(self):
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(linestyle='--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim(0, self.lattice_y)
        plt.xlim(0, self.lattice_x)
        for qudit in self.qudit_list:
            x1 = qudit.coordinates[0]['x']
            x2 = qudit.coordinates[1]['x']
            y1 = qudit.coordinates[0]['y']
            y2 = qudit.coordinates[1]['y']
            plt.plot([x1, x2], [y1, y2], linestyle = ':', color='blue')
            scatter_x_list = []
            scatter_y_list = [] 
            for instance in qudit.instances_positions:
                scatter_x_list.append(instance['x'])
                scatter_y_list.append(instance['y'])
            plt.scatter(scatter_x_list, scatter_y_list, marker='o', color='black', s= 10*qudit.mass/(1e-14))
        #plt.savefig('test.png')
        plt.show()


class Qudit:
    def __init__(self, dimension, coordinates, mass):
        self.coordinates = coordinates
        instances_positions = []
        start_x = coordinates[0]['x']
        start_y = coordinates[0]['y']
        end_x = coordinates[1]['x']
        end_y = coordinates[1]['y']
        x_length = np.abs(coordinates[0]['x'] - coordinates[1]['x'])
        y_length = np.abs(coordinates[0]['y'] - coordinates[1]['y'])
        for instance in range(dimension):
            instances_positions.append({'x': start_x + (end_x - start_x)*instance/(dimension -1), 
                                        'y': start_y + (end_y - start_y)*instance/(dimension -1)}) 
                                        
        self.dimension = dimension
        self.delta_x = np.sqrt((coordinates[0]['x'] - coordinates[1]['x'])**2
                              +(coordinates[0]['y'] - coordinates[1]['y'])**2)
        self.instances_positions = instances_positions
        self.mass = mass


class Quantum_State:
    def __init__(self, qudit_list, time, physics_values, gamma= 0.0):
        self.qudit_list = qudit_list
        self.time = time
        self.physics_values = physics_values
        self.gamma = gamma

    def state_dimentions(self):
        dim_list = []
        for qudit in self.qudit_list:
            dim_list.append(qudit.dimension)
        return dim_list

    def vector(self):
        num_qudits = len(self.qudit_list)
        instances_indices = np.zeros((num_qudits))
        vector_length = 1
        for qudit in self.qudit_list: 
            vector_length = vector_length*qudit.dimension
        norm_factor = np.sqrt(vector_length)
        vector = []
        for entry in range(vector_length):
            instance_list = []
            phase_factor = 0
            for count, qudit in enumerate(self.qudit_list):
                instance_list.append(qudit.instances_positions[int(instances_indices[count])])
            if entry < vector_length - 1:
                instances_indices = self.update_indices_list(instances_indices)
            for count_1, instance_1 in enumerate(instance_list[0: -1]):
                for count_2, instance_2 in enumerate(instance_list[count_1 + 1:]):
                    mass_1 = self.qudit_list[count_1].mass
                    mass_2 = self.qudit_list[count_1 + count_2 + 1].mass
                    distance = self.instances_distance(instance_1, instance_2)
                    phase_factor += (mass_1 * mass_2) / distance
            phase_factor = (self.physics_values.gravity/self.physics_values.h_bar)*phase_factor
            phase = np.exp(1.0j*phase_factor*self.time)
            vector.append(phase)
        return (1/norm_factor)*np.array([vector])

    def update_indices_list(self, instances_indices):
        indices_update_pending = True
        last_index = 1
        while indices_update_pending:
            instances_indices[-last_index] += 1
            if instances_indices[-last_index] == self.qudit_list[-last_index].dimension:
                instances_indices[-last_index] = 0
                last_index += 1
            else: 	
                indices_update_pending = False
        return instances_indices

    def instances_distance(self, instance_1, instance_2):
        x_difference = np.abs(instance_1['x'] - instance_2['x'])
        y_difference = np.abs(instance_1['y'] - instance_2['y'])
        distance = np.sqrt(x_difference**2 + y_difference**2)
        return distance
      
    def density_matrix(self):
        """ Return the density matrix of the state, without taking into account decoherence"""
        state_vector = self.vector()
        density_matrix = np.kron(state_vector, state_vector.conj().T)
        return density_matrix

    def density_matrix_decohere_old(self, physics_values): #TODO
        """ Return the density matrix of the state, taking into account decoherence"""
        state_vector = self.vector(physics_values)
        decoherence_mask = np.identity(self.dimension)
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i != j:
                    decoherence_mask[i][j] = np.exp(-1.*self.gamma*self.time)
        decoherence_mask = np.kron(decoherence_mask, decoherence_mask)
        density_matrix = np.kron(state_vector, state_vector.conj().T) * decoherence_mask
        return density_matrix

    def density_matrix_decohere(self):
        """ Return the density matrix of the state, taking into account decoherence"""
        state_vector = self.vector()
        first = True
        for qudit in self.qudit_list:
            qudit_deco_mask = np.identity(qudit.dimension)
            for i in range(qudit.dimension):
                for j in range(qudit.dimension):
                    if i != j:
                        qudit_deco_mask[i][j] = np.exp(-1.*self.gamma*self.time)
            if first:
                decoherence_mask = qudit_deco_mask
            else:             
                decoherence_mask = np.kron(decoherence_mask, qudit_deco_mask)
            first = False
        density_matrix = np.kron(state_vector, state_vector.conj().T) * decoherence_mask
        return density_matrix

    def reduced_density_matrix(self):
        """ Return the density matrix of the state, without taking into account decoherence"""
        density_mat = self.density_matrix()
        density_mat_size = density_mat.shape[0]
        ref_system = self.qudit_list[0]
        alt_system_dimension = int(density_mat_size / ref_system.dimension)
        partial_trace = np.ndarray(shape=(ref_system.dimension , ref_system.dimension), dtype=np.complex_)
        for row_block in range(0, ref_system.dimension):
            for column_block in range(0, ref_system.dimension):
                mat_element = 0
                for i in range(row_block*alt_system_dimension, (row_block + 1)*alt_system_dimension):
                    for j in range(column_block*alt_system_dimension, (column_block+1)*alt_system_dimension):
                        if i%alt_system_dimension == j%alt_system_dimension:
                            mat_element = mat_element + density_mat[i][j] 
                partial_trace[row_block][column_block] = mat_element   
        return partial_trace


    def von_neuman_entropy(self):
        """ Compute bipartite VNE for the state speficied """
        density_mat = self.density_matrix()
        density_mat_size = density_mat.shape[0]
        ref_system = self.qudit_list[0]
        alt_system_dimension = int(density_mat_size / ref_system.dimension)
        partial_trace = np.ndarray(shape=(ref_system.dimension , ref_system.dimension), dtype=np.complex_)
        for row_block in range(0, ref_system.dimension):
            for column_block in range(0, ref_system.dimension):
                mat_element = 0
                for i in range(row_block*alt_system_dimension, (row_block + 1)*alt_system_dimension):
                    for j in range(column_block*alt_system_dimension, (column_block+1)*alt_system_dimension):
                        if i%alt_system_dimension == j%alt_system_dimension:
                            mat_element = mat_element + density_mat[i][j] 
                partial_trace[row_block][column_block] = mat_element
        evalsh, evectsh = np.linalg.eigh(partial_trace)
        vn_entropy = 0
        for value in evalsh:
            if value > 0: 
                vn_entropy = vn_entropy - value*np.log(value)
        return vn_entropy

    @staticmethod
    def _partial_transpose_dense(rho, mask, dimensions_list):
        """
        Based on Jonas' implementation using numpy.
        Very fast for dense problems. """

        dims = [dimensions_list, dimensions_list]
        nsys = len(mask)
        pt_dims = np.arange(2 * nsys).reshape(2, nsys).T
        pt_idx = np.concatenate([[pt_dims[n, mask[n]] for n in range(nsys)],
                                [pt_dims[n, 1 - mask[n]] for n in range(nsys)]])

        partial_transpose = rho.reshape(
            np.array(dims).flatten()).transpose(pt_idx).reshape(rho.shape)

        return partial_transpose


    def find_PPT_EW(self, physics_values, mask):
        """ Returns the matrix form of the PPT entanglement witness for a given state """
        num_qudits = len(self.qudit_list)
        dimensions_list =[]
        for qudit in self.qudit_list:
            dimensions_list.append(qudit.dimension)
        pt_mat = Quantum_State._partial_transpose_dense(self.density_matrix(), mask, dimensions_list)
        evals, evects = np.linalg.eigh(pt_mat)
        min_eval = np.where(evals == np.amin(evals))
        vect = evects[:,min_eval]
        dimensions_prod = 1
        for dim in dimensions_list:
            dimensions_prod = dimensions_prod*dim
        vect = evects[:,min_eval].reshape(dimensions_prod,1)
        if evals[min_eval] <-1e-15:
            eigenstate_density = np.kron(vect, vect.conj().T)
            witness = Quantum_State._partial_transpose_dense(eigenstate_density, mask, dimensions_list)
            return witness 
        else:
            return False



class EW:
    """ Defines an Entanglement Witness, the matrix form of which must be specified at 
        initialisation"""
    def __init__(self, matrix_rep):
        self.matrix_rep = matrix_rep

    def apply_to_state(self, q_state, physics_values):
        """ Returns the expectation value of the witness with respect to a given quantum state"""
        #density_mat = q_state.density_matrix_decohere(physics_values)
        density_mat = q_state.density_matrix_decohere()
        return np.real(np.trace(np.matmul(self.matrix_rep, density_mat)))

    def decomposition(self, basis):
        """ Returns the decomposition of the witness in terms of Pauli strings and weights"""
        weights = []
        dimension = float(self.matrix_rep.shape[0])
        for operator in basis:
               weight = np.trace(np.matmul(self.matrix_rep, operator))
               if abs(weight) < 1e-12:
                   weight = 0
               weights.append(weight)
        return (1./dimension)*np.array(weights)




def build_gate(string):
    Identity = np.array([[1.0, 0.0], [0.0, 1.0]])
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    H = (1.0/np.sqrt(2))*np.array([[1.0, 1.0], [1.0, -1.0]])
    if string[0] == 'I':
        gate = Identity
    if string[0] == 'X':
        gate = X
    if string[0] == 'Y':
        gate = Y
    if string[0] == 'Z':
        gate = Z
    if string[0] == 'H':
        gate = H
    for letter_id in range(1, len(string)):
        if string[letter_id] == 'I':
            gate = np.kron(gate, Identity)
        if string[letter_id] == 'X':
            gate = np.kron(gate, X)
        if string[letter_id] == 'Y':
            gate = np.kron(gate, Y)
        if string[letter_id] == 'Z':
            gate = np.kron(gate, Z)
        if string[letter_id] == 'H':
            gate = np.kron(gate, H)
    return gate
        
physics_values = Physics(6.67408*1e-11, 1.054718*1e-34)



####Step 1 - create a few qudits, each are created as follows:
## qudit_1 = Qudit(dimension, [coordinate for leftmost superposition instance], [coordinate for rightmost superposition instance]], mass)

####Step 2 - create a lattice on which the model is run (it's actually just for creating plots!):
## lattice_test = Model_Lattice(lattice_x_dimension, lattice_y_dimension)

####Step 3 - add the qudits to the lattice:
## lattice_test.add_qudit(qudit_1)
## lattice_test.add_qudit(qudit_2)

####Step 4 - Generate the quantum state:
## quantum_state = Quantum_State(lattice_test.qudit_list, time, physics_values)

####Step 5 - Compute VNE:
## VNE = quantum_state.von_neuman_entropy()


######### Below are a few examples of different set-up you can create (check the attached pdf for the corresponding plot #####

## 3 box
#qudit_1 = Qudit(2, [{'x':300*1e-6 + 75*1e-6, 'y':100*1e-6 + 150*1e-6}, {'x':550*1e-6+75*1e-6, 'y':100*1e-6+ 150*1e-6}], 1e-14)
#qudit_2 = Qudit(2, [{'x':200*1e-6+75*1e-6, 'y':273.205*1e-6+ 150*1e-6}, {'x':325*1e-6+75*1e-6, 'y':489.71*1e-6+ 150*1e-6}], 1e-14)
#qudit_3 = Qudit(2, [{'x':650*1e-6+75*1e-6, 'y':273.205*1e-6+ 150*1e-6}, {'x':525*1e-6+75*1e-6, 'y':489.71*1e-6+ 150*1e-6}], 1e-14)

## 3 star
#qudit_1 = Qudit(2, [{'x':500*1e-6, 'y':100*1e-6 + 110*1e-6 }, {'x':500*1e-6, 'y':350*1e-6 + 110*1e-6 }], 1e-14)
#qudit_2 = Qudit(2, [{'x':400*1e-6, 'y':523.205*1e-6  + 110*1e-6 }, {'x':183.493677*1e-6, 'y':648.205*1e-6 + 110*1e-6 }], 1e-14)
#qudit_3 = Qudit(2, [{'x':600*1e-6, 'y':523.205*1e-6 + 110*1e-6 }, {'x':816.506322896*1e-6, 'y':648.205*1e-6 + 110*1e-6 }], 1e-14)

# 3 rectangle
#qudit_1 = Qudit(2, [{'x':375*1e-6, 'y':175*1e-6}, {'x':625*1e-6, 'y':175*1e-6}], 1e-14)
#qudit_2 = Qudit(2, [{'x':500*1e-6, 'y':375*1e-6}, {'x':500*1e-6, 'y':625*1e-6}], 1e-14)
#qudit_3 = Qudit(2, [{'x':375*1e-6, 'y':825*1e-6}, {'x':625*1e-6, 'y':825*1e-6}], 1e-14)


#4 star
#qudit_1 = Qudit(2, [{'x':500*1e-6, 'y':100*1e-6}, {'x':500*1e-6, 'y':350*1e-6}], 1e-14)
#qudit_2 = Qudit(2, [{'x':358.58*1e-6, 'y':491.42*1e-6}, {'x':108*1e-6, 'y':491.42*1e-6}], 1e-14)
#qudit_3 = Qudit(2, [{'x':641.42*1e-6, 'y':491.42*1e-6}, {'x':891.42*1e-6, 'y':491.42*1e-6}], 1e-14)
#qudit_4 = Qudit(2, [{'x':500*1e-6, 'y':632.84*1e-6}, {'x':500*1e-6, 'y':882.84*1e-6}], 1e-14)

# 4 box
#qudit_1 = Qudit(2, [{'x':375*1e-6, 'y':233.58*1e-6}, {'x':625*1e-6, 'y':233.58*1e-6}], 1e-14)
#qudit_2 = Qudit(2, [{'x':233.58*1e-6, 'y':375*1e-6}, {'x':233.58*1e-6, 'y':625*1e-6}], 1e-14)
#qudit_3 = Qudit(2, [{'x':375*1e-6, 'y':766.42*1e-6}, {'x':625*1e-6, 'y':766.42*1e-6}], 1e-14)
#qudit_4 = Qudit(2, [{'x':766.42*1e-6, 'y':375*1e-6}, {'x':766.42*1e-6, 'y':625*1e-6}], 1e-14)

# 4 rectangle
#qudit_1 = Qudit(2, [{'x':375*1e-6, 'y':175*1e-6}, {'x':625*1e-6, 'y':175*1e-6}], 1e-14)
#qudit_2 = Qudit(2, [{'x':375*1e-6, 'y':375*1e-6}, {'x':375*1e-6, 'y':625*1e-6}], 1e-14)
#qudit_3 = Qudit(2, [{'x':375*1e-6, 'y':825*1e-6}, {'x':625*1e-6, 'y':825*1e-6}], 1e-14)
#qudit_4 = Qudit(2, [{'x':625*1e-6, 'y':375*1e-6}, {'x':625*1e-6, 'y':625*1e-6}], 1e-14)

# Parallel 3
qudit_1 = Qudit(2, [{'x':400*1e-6, 'y':375*1e-6}, {'x':400*1e-6, 'y':625*1e-6}], 1e-14)
qudit_2 = Qudit(2, [{'x':600*1e-6, 'y':375*1e-6}, {'x':600*1e-6, 'y':625*1e-6}], 1e-14)
qudit_3 = Qudit(2, [{'x':800*1e-6, 'y':375*1e-6}, {'x':800*1e-6, 'y':625*1e-6}], 1e-14)
#qudit_4 = Qudit(3, [{'x':1000*1e-6, 'y':375*1e-6}, {'x':1000*1e-6, 'y':625*1e-6}], 1e-14)

# Parallel 2
qudit_4 = Qudit(2, [{'x':400*1e-6, 'y':375*1e-6}, {'x':400*1e-6, 'y':625*1e-6}], 1e-14)
qudit_5 = Qudit(2, [{'x':600*1e-6, 'y':375*1e-6}, {'x':600*1e-6, 'y':625*1e-6}], 1e-14)

# Parallel 4
qudit_6 = Qudit(2, [{'x':400*1e-6, 'y':375*1e-6}, {'x':400*1e-6, 'y':625*1e-6}], 1e-14)
qudit_7 = Qudit(2, [{'x':600*1e-6, 'y':375*1e-6}, {'x':600*1e-6, 'y':625*1e-6}], 1e-14)
qudit_8 = Qudit(2, [{'x':800*1e-6, 'y':375*1e-6}, {'x':800*1e-6, 'y':625*1e-6}], 1e-14)
qudit_9 = Qudit(2, [{'x':1000*1e-6, 'y':375*1e-6}, {'x':1000*1e-6, 'y':625*1e-6}], 1e-14)

# Parallel 5
qudit_10 = Qudit(2, [{'x':400*1e-6, 'y':375*1e-6}, {'x':400*1e-6, 'y':625*1e-6}], 1e-14)
qudit_11 = Qudit(2, [{'x':600*1e-6, 'y':375*1e-6}, {'x':600*1e-6, 'y':625*1e-6}], 1e-14)
qudit_12 = Qudit(2, [{'x':800*1e-6, 'y':375*1e-6}, {'x':800*1e-6, 'y':625*1e-6}], 1e-14)
qudit_13 = Qudit(2, [{'x':1000*1e-6, 'y':375*1e-6}, {'x':1000*1e-6, 'y':625*1e-6}], 1e-14)
qudit_14 = Qudit(2, [{'x':1200*1e-6, 'y':375*1e-6}, {'x':1200*1e-6, 'y':625*1e-6}], 1e-14)

# Parallel 6
qudit_15 = Qudit(2, [{'x':400*1e-6, 'y':375*1e-6}, {'x':400*1e-6, 'y':625*1e-6}], 1e-14)
qudit_16 = Qudit(2, [{'x':600*1e-6, 'y':375*1e-6}, {'x':600*1e-6, 'y':625*1e-6}], 1e-14)
qudit_17 = Qudit(2, [{'x':800*1e-6, 'y':375*1e-6}, {'x':800*1e-6, 'y':625*1e-6}], 1e-14)
qudit_18 = Qudit(2, [{'x':1000*1e-6, 'y':375*1e-6}, {'x':1000*1e-6, 'y':625*1e-6}], 1e-14)
qudit_19 = Qudit(2, [{'x':1200*1e-6, 'y':375*1e-6}, {'x':1200*1e-6, 'y':625*1e-6}], 1e-14)
qudit_20 = Qudit(2, [{'x':1400*1e-6, 'y':375*1e-6}, {'x':1400*1e-6, 'y':625*1e-6}], 1e-14)

# Playground:
#qudit_1 = Qudit(2, [{'x':200*1e-6, 'y':200*1e-6}, {'x':450*1e-6, 'y':200*1e-6}], 1e-14)
#qudit_2 = Qudit(2, [{'x':200*1e-6, 'y':400*1e-6}, {'x':450*1e-6, 'y':400*1e-6}], 1e-14)
#qudit_3 = Qudit(2, [{'x':200*1e-6, 'y':600*1e-6}, {'x':450*1e-6, 'y':600*1e-6}], 1e-14)
#qudit_4 = Qudit(2, [{'x':800*1e-6, 'y':375*1e-6}, {'x':800*1e-6, 'y':625*1e-6}], 1e-14)

# Playground:
#qudit_1 = Qudit(2, [{'x':100*1e-6, 'y':500*1e-6}, {'x':300*1e-6, 'y':500*1e-6}], 1e-14)
#qudit_2 = Qudit(2, [{'x':550*1e-6, 'y':500*1e-6}, {'x':750*1e-6, 'y':500*1e-6}], 1e-14)
#qudit_3 = Qudit(2, [{'x':1000*1e-6, 'y':500*1e-6}, {'x':1200*1e-6, 'y':500*1e-6}], 1e-14)
#qudit_4 = Qudit(2, [{'x':800*1e-6, 'y':375*1e-6}, {'x':800*1e-6, 'y':625*1e-6}], 1e-14)





### Weight change
#m2 = 1e-14
#m1 = 10*1e-14
#d_x_1 = (1.604*1e-15 / m1)*1e-6*2000
#d_x_2 = (1.604*1e-15 / m2)*1e-6*2000

#centre_y = 500*1e-6
#qudit_1 = Qudit(2, [{'x':400*1e-6, 'y':centre_y - d_x_1/2}, {'x':400*1e-6, 'y':centre_y + d_x_1/2}], m1)
#qudit_2 = Qudit(2, [{'x':600*1e-6, 'y':centre_y - d_x_2/2}, {'x':600*1e-6, 'y':centre_y + d_x_2/2}], m2)
#qudit_3 = Qudit(2, [{'x':800*1e-6, 'y':centre_y - d_x_1/2}, {'x':800*1e-6, 'y':centre_y + d_x_1/2}], m1)
#qudit_4 = Qudit(2, [{'x':200*1e-6, 'y':centre_y - d_x_2/2}, {'x':200*1e-6, 'y':centre_y + d_x_2/2}], m2)

lattice_test_sext = Model_Lattice(1500*1e-6, 1500*1e-6)
lattice_test_sext.add_qudit(qudit_15)
lattice_test_sext.add_qudit(qudit_16)
lattice_test_sext.add_qudit(qudit_17)
lattice_test_sext.add_qudit(qudit_18)
lattice_test_sext.add_qudit(qudit_19)
lattice_test_sext.add_qudit(qudit_20)
quantum_state_sext = Quantum_State(lattice_test_sext.qudit_list, 2.5, physics_values)

lattice_test_quin = Model_Lattice(1500*1e-6, 1500*1e-6)
lattice_test_quin.add_qudit(qudit_10)
lattice_test_quin.add_qudit(qudit_11)
lattice_test_quin.add_qudit(qudit_12)
lattice_test_quin.add_qudit(qudit_13)
lattice_test_quin.add_qudit(qudit_14)
quantum_state_quin = Quantum_State(lattice_test_quin.qudit_list, 2.5, physics_values)

lattice_test_quad = Model_Lattice(1500*1e-6, 1500*1e-6)
lattice_test_quad.add_qudit(qudit_6)
lattice_test_quad.add_qudit(qudit_7)
lattice_test_quad.add_qudit(qudit_8)
lattice_test_quad.add_qudit(qudit_9)
quantum_state_quad = Quantum_State(lattice_test_quad.qudit_list, 2.5, physics_values)

lattice_test_triple = Model_Lattice(1500*1e-6, 1500*1e-6)
lattice_test_triple.add_qudit(qudit_1)
lattice_test_triple.add_qudit(qudit_2)
lattice_test_triple.add_qudit(qudit_3)
quantum_state_triple = Quantum_State(lattice_test_triple.qudit_list, 2.5, physics_values)

lattice_test_double = Model_Lattice(1500*1e-6, 1500*1e-6)
lattice_test_double.add_qudit(qudit_4)
lattice_test_double.add_qudit(qudit_5)
quantum_state_double = Quantum_State(lattice_test_double.qudit_list, 2.5, physics_values)


#GHZ = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) 
#GHZ = (1/np.sqrt(2))*GHZ

#IIX = build_gate('III')
#GHZ = np.dot(IIX, GHZ.T)

#gate = build_gate('III')
#GHZ = np.dot(IIX, GHZ.T)
#quantum_state.time = 13.0
#state =np.dot(gate, quantum_state.vector()[0])
#print(np.abs(np.dot(GHZ.conj(), state)**2))

#for t in range(60):
#    quantum_state.time = t
#    print('Time: ', t)
#    print('Overlap: ', np.abs(np.dot(GHZ.T.conj(), quantum_state.vector())[0][0]))
#    print('VNE: ', quantum_state.von_neuman_entropy())


#print(quantum_state.vector())
#print("Von Neuman entropy (partial trace on first qudit): ", quantum_state.von_neuman_entropy())

mask_sext = [1, 0, 1, 0, 1, 0] #from Matlab code has highest entropy
mask_quin = [1, 0, 1, 0, 1] #from Matlab code has highest entropy
mask_quad = [1, 0, 1, 0] #from Matlab code has highest entropy
mask_triple =  [0, 1, 0] #select the qudit(s) which are partial transposed
mask_double =  [0, 1] #select the qudit(s) which are partial transposed


##Counting the operators:
# 6 qubits case:
paulis = all_letters(6)
basis = gen_paulis_tensors(2, 6)
EW_PPT_sext = EW(quantum_state_sext.find_PPT_EW(physics_values, mask_sext))
weight_sext = EW_PPT_sext.decomposition(basis)
pauli_to_measure_sext = [paulis[i] for i in range(len(weight_sext)) if weight_sext[i] != 0]
count_paulis_sext = len(pauli_to_measure_sext)
groups_sext = tpb_group(pauli_to_measure_sext)

# 5 qubits case:
paulis = all_letters(5)
basis = gen_paulis_tensors(2, 5)
EW_PPT_quin = EW(quantum_state_quin.find_PPT_EW(physics_values, mask_quin))
weight_quin = EW_PPT_quin.decomposition(basis)
pauli_to_measure_quin = [paulis[i] for i in range(len(weight_quin)) if weight_quin[i] != 0]
count_paulis_quin = len(pauli_to_measure_quin)
groups_quin = tpb_group(pauli_to_measure_quin)

# 4 qubits case:
paulis = all_letters(4)
basis = gen_paulis_tensors(2, 4)
EW_PPT_quad = EW(quantum_state_quad.find_PPT_EW(physics_values, mask_quad))
weight_quad = EW_PPT_quad.decomposition(basis)
pauli_to_measure_quad = [paulis[i] for i in range(len(weight_quad)) if weight_quad[i] != 0]
count_paulis_quad = len(pauli_to_measure_quad)
groups_quad = tpb_group(pauli_to_measure_quad)

# 3 qubits case:
paulis = all_letters(3)
basis = gen_paulis_tensors(2, 3)
EW_PPT_triple = EW(quantum_state_triple.find_PPT_EW(physics_values, mask_triple))
weight_triple = EW_PPT_triple.decomposition(basis)
pauli_to_measure_triple = [paulis[i] for i in range(len(weight_triple)) if weight_triple[i] != 0]
count_paulis_triple = len(pauli_to_measure_triple)
groups_triple = tpb_group(pauli_to_measure_triple)

# 2 qubits case:
paulis = all_letters(2)
basis = gen_paulis_tensors(2, 2)
EW_PPT_double = EW(quantum_state_double.find_PPT_EW(physics_values, mask_double))
weight_double = EW_PPT_double.decomposition(basis)
pauli_to_measure_double = [paulis[i] for i in range(len(weight_double)) if weight_double[i] != 0]
count_paulis_double = len(pauli_to_measure_double)
groups_double = tpb_group(pauli_to_measure_double)

        
## Decoherence test:
gamma_list = np.arange(0.02, 0.2, 0.01)
sext_list = []
quin_list = []
quad_list = []
triple_list = []
double_list = []
for gamma in gamma_list:
    quantum_state_sext.gamma = gamma
    quantum_state_quin.gamma = gamma
    quantum_state_quad.gamma = gamma
    quantum_state_triple.gamma = gamma
    quantum_state_double.gamma = gamma
    #
    EW_PPT_sext = EW(quantum_state_sext.find_PPT_EW(physics_values, mask_sext))
    EW_PPT_quin = EW(quantum_state_quin.find_PPT_EW(physics_values, mask_quin))
    EW_PPT_quad = EW(quantum_state_quad.find_PPT_EW(physics_values, mask_quad))
    EW_PPT_triple = EW(quantum_state_triple.find_PPT_EW(physics_values, mask_triple))
    EW_PPT_double = EW(quantum_state_double.find_PPT_EW(physics_values, mask_double))
    #
    expect_sext = EW_PPT_sext.apply_to_state(quantum_state_sext, physics_values)
    expect_quin = EW_PPT_quin.apply_to_state(quantum_state_quin, physics_values)
    expect_quad = EW_PPT_quad.apply_to_state(quantum_state_quad, physics_values)
    expect_triple = EW_PPT_triple.apply_to_state(quantum_state_triple, physics_values) 
    expect_double = EW_PPT_double.apply_to_state(quantum_state_double, physics_values)
    #
    sext_list.append(expect_sext)
    quin_list.append(expect_quin)
    quad_list.append(expect_quad)
    triple_list.append(expect_triple)
    double_list.append(expect_double)

## Decoherence time:
sext_listt = []
quin_listt = []
quad_listt = []
triple_listt = []
double_listt = []
gamma = 0.16
for t in np.arange(0.5,5,0.25):
    quantum_state_sext.gamma = gamma
    quantum_state_quin.gamma = gamma
    quantum_state_quad.gamma = gamma
    quantum_state_triple.gamma = gamma
    quantum_state_double.gamma = gamma
    #
    quantum_state_sext.time = t
    quantum_state_quin.time = t
    quantum_state_quad.time = t
    quantum_state_triple.time = t
    quantum_state_double.time = t
    #
    EW_PPT_sext = EW(quantum_state_sext.find_PPT_EW(physics_values, mask_sext))
    EW_PPT_quin = EW(quantum_state_quin.find_PPT_EW(physics_values, mask_quin))
    EW_PPT_quad = EW(quantum_state_quad.find_PPT_EW(physics_values, mask_quad))
    EW_PPT_triple = EW(quantum_state_triple.find_PPT_EW(physics_values, mask_triple))
    EW_PPT_double = EW(quantum_state_double.find_PPT_EW(physics_values, mask_double))
    #
    expect_sext = EW_PPT_sext.apply_to_state(quantum_state_sext, physics_values)
    expect_quin = EW_PPT_quin.apply_to_state(quantum_state_quin, physics_values)
    expect_quad = EW_PPT_quad.apply_to_state(quantum_state_quad, physics_values)
    expect_triple = EW_PPT_triple.apply_to_state(quantum_state_triple, physics_values) 
    expect_double = EW_PPT_double.apply_to_state(quantum_state_double, physics_values)
    #
    sext_listt.append(expect_sext)
    quin_listt.append(expect_quin)
    quad_listt.append(expect_quad)
    triple_listt.append(expect_triple)
    double_listt.append(expect_double)


## Results print-out:

print('---Results are provisional until fully tested!---')
print('\n')
print('-- List of operators to measure (Identity does not count): ')
print('Two-qubit case:   ', pauli_to_measure_double)
print('         Count:   ', count_paulis_double)
print('Three-qubit case: ', pauli_to_measure_triple)
print('         Count:   ', count_paulis_triple)
print('Four-qubit case: ', pauli_to_measure_quad)
print('         Count:   ', count_paulis_quad)
print('\n')
print('-- Tensor Product Basis groupins: ')
print('Two-qubit case:   ', groups_double)
print('         Count:', len(groups_double))
print('Three-qubit case: ', groups_triple)
print('           Count:', len(groups_triple))
print('Four-qubit case: ', groups_quad)
print('          Count:', len(groups_quad))
print('Five-qubit case: ', groups_quin)
print('          Count:', len(groups_quin))
print('Six-qubit case: ', groups_sext)
print('          Count:', len(groups_sext))
print('\n')
print('-- Decoherence test (EW expectation): ')
print('Gamma rates tested:    ', gamma_list)
print('Two-qubit EW expect:   ', double_list)
print('Three-qubit EW expect: ', triple_list)
print('Four-qubit EW expect: ', quad_list)
print('\n')

## Results plots:

plt.plot(np.arange(0.5,5,0.25), double_listt, '.-', label='n=2')
plt.plot(np.arange(0.5,5,0.25), triple_listt, '.-', label='n=3')
plt.plot(np.arange(0.5,5,0.25), quad_listt, '.-', label='n=4')
plt.plot(np.arange(0.5,5,0.25), quin_listt, '.-', label='n=5')
plt.plot(np.arange(0.5,5,0.25), sext_listt, '.-', label='n=6')
plt.ylabel('EW expectation value')
plt.xlabel('time [s]')
plt.title('PPT Witness expectation value over time with gamma=0.15')
plt.legend()
plt.show()

plt.plot(gamma_list, double_list, '.-', label='2 particles')
plt.plot(gamma_list, triple_list, '.-', label='3 particles')
plt.plot(gamma_list, quad_list, '.-', label='4 particles')
plt.plot(gamma_list, quin_list, '.-', label='5 particles')
plt.plot(gamma_list, sext_list, '.-', label='6 particles')
plt.xlabel('gamma rates')
plt.ylabel('EW expectation value')
plt.title('PPT Witness expectation value for different decoherence rates')
plt.legend()
plt.show()

plt.plot([2,3,4,5,6], [len(groups_double),len(groups_triple),len(groups_quad),len(groups_quin),len(groups_sext)], '.-')
plt.xlabel('# of superpositions')
plt.ylabel('# of measurements needed')
plt.title('Number of measurements needed (PPT Witness) for different system sizes')
plt.show()

#lattice_test.plot_lattice()


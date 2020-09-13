"""
A module for working with the Quantum Harmonic Oscillator.
@author Marco, marco.bizzness@gmail.com
"""

import numpy as np
from scipy import special

iterable = list or tuple or set or np.array  # custom iterable type


class QuantumOscillator:
    def __init__(self, n_list: iterable, c_list: iterable, omega=1, mass=1, h_bar=1, name=" "):
        """
        Class that describes a generic Quantum Harmonic Oscillator state.
        H = 1/2m * p^2  +  1/2 * m * omega^2 * x^2  (Hamiltonian)
        Initialize a dictionary where:
        key = n
        value = vector with all elements set to 0, except for the i-th element
        which is equal to c_array coefficient divided by c_array's norm.
        :parameter n_list: quantum numbers list, will define various |n> states
        :parameter c_list: coefficients list of |n> states (not necessarily normalized, we'll do that for you :) )
        """
        # ket name
        self.name = name

        # physical parameters
        self.mass = mass
        self.omega = omega
        self.h_bar = h_bar

        self.alpha = np.sqrt(self.mass * self.omega / self.h_bar)  # useful for wavefunction

        # Hilbert Space parameters
        self.dim = len(n_list)
        self.c_list = [c for n, c in sorted(zip(n_list, c_list))]
        self.n_list = sorted(n_list)

        # private parameters used for defining our Hilbert Space ket
        self._norm = np.linalg.norm(self.c_list)
        self._n_vector = np.identity(self.dim)

        # Hilbert Space ket in dictionary form (key = n, value = vector)
        self.n_state = {n: (self.c_list[i] * self._n_vector[i]) / self._norm for (i, n) in enumerate(self.n_list)}

        # Hilbert Space in dictionary form (key = n, value = coefficient)
        self.n_coeffs = {n: self.c_list[i] / self._norm for (i, n) in enumerate(self.n_list)}

        # Hilbert Space ket in array form
        self.state_vector = sum(self.n_state.values())

        # renormalization (purely physical, useless for code purposes)
        self._norm = 1

    def _number(self):
        """
        Does NOT apply number operator.
        :returns mean value of number operator
        """
        return sum([n * modulus_squared(v) for (n, v) in self.n_coeffs.items()])

    def _remove_zero(self):
        """
        If |0> is part of the state, removes it and shortens array in self.n_states by 1.
        """
        try:
            del self.n_state[0]
            del self.n_coeffs[0]
            self.dim -= 1
            self.n_state = {n: np.delete(self.n_state[n], 0) for n in self.n_state.keys()}
        except KeyError:
            pass

    def _apply_operator_util(self, n_shift, result_shift, norm, number=False):
        """
        Utility function when applying operators
        """
        # in place state modification
        if not number:
            self.n_state = {n + n_shift:
                            (np.sqrt(n + result_shift) * self.n_state[n]) / norm for n in self.n_state.keys()}
            self.n_coeffs = {n + n_shift:
                             (np.sqrt(n + result_shift) * self.n_coeffs[n]) / norm for n in self.n_coeffs.keys()}
        else:
            self.n_state = {n: (n * self.n_state[n]) / self._norm for n in self.n_state.keys()}
            self.n_coeffs = {n: (n * self.n_coeffs[n]) / self._norm for n in self.n_coeffs.keys()}

        self.state_vector = sum(self.n_state.values())

        # renormalization (purely physical, useless for code purposes)
        self._norm = 1

    def number(self):
        """
        Applies number operator to state.
        :returns <n| N |n> mean value of number operator
        """
        self._norm = return_value = self._number()

        # try removing |0>
        self._remove_zero()

        # apply operator and modify state
        self._apply_operator_util(n_shift=0, result_shift=0, norm=self._norm, number=True)

        return return_value

    def annihilation(self):
        """
        Applies annihilation operator to state.
        :returns annihilation result ( sqrt(number operator) )
        """
        self._norm = return_value = np.sqrt(self._number())

        # try removing |0>
        self._remove_zero()

        # apply operator and modify state
        self._apply_operator_util(n_shift=-1, result_shift=0, norm=self._norm)

        return return_value

    def creation(self):
        """
        Applies creation operator to state.
        :returns creation result ( sqrt(number operator + 1) )
        """
        self._norm = return_value = np.sqrt(self._number() + 1)

        # apply operator and modify state
        self._apply_operator_util(n_shift=1, result_shift=1, norm=self._norm)

        return return_value

    def hamiltonian(self):
        """
        Applies Hamiltonian H.
        :returns <n| H |n> mean value of energy
        """
        return (self.number() + 0.5) * self.h_bar * self.omega

    def number_mean(self):
        """
        :returns <n| N |n> mean value of number operator
        """
        return self._number()

    def energy_mean(self):
        """
        Does NOT apply Hamiltonian.
        :returns <n| H |n> mean value of energy
        """
        return (self._number() + 0.5) * self.h_bar * self.omega

    def position_squared_mean(self):
        """
        :returns <n| x^2 |n> mean position squared value
        """
        return (self._number() + 0.5) * self.h_bar / (self.mass * self.omega)

    def momentum_squared_mean(self):
        """
        :returns <n| p^2 |n> mean position squared value
        """
        return self.energy_mean() * self.mass

    def n_probability(self, n):
        """
        :parameter n: Harmonic Oscillator quantum number
        :returns probability of finding system in state |n>
        """
        if n in self.n_state.keys():
            return modulus_squared(self.n_coeffs[n])
        else:
            return 0

    def ket(self):
        """
        Prints ket state with respect to various |n> kets
        """
        ket_string = ""
        for n, c in self.n_coeffs.items():
            ket_string += f'{np.around(c, 3)}|{n}> + '
        ket_string = ket_string[:-3]
        print(f'\n|{self.name}> = {ket_string}\n')

    def display(self):
        """
        Prints:
        omega, h_bar, energy (without applying Hamiltonian to state, thus not removing any possible |0> state)
        and state in ket |n> notation.
        """
        print(f'omega={self.omega} \t h_bar={self.h_bar}')
        print(f'<E> = h_bar * omega * (<n> + 1/2) = {self.energy_mean()}')

        self.ket()

    def eval_wavefunction(self, x_array: iterable):
        """
        Computes wave function, based on eigenfunctions psi_n(x) = A_n * H_n(alpha * x) * exp(-alpha^2 * x^2 / 2)
        :parameter x_array: array containing points on the x-position axis
        :returns dictionary of two arrays containing wavefunction's real and imaginary parts evaluated in x_array
        key = 'real' : real part
        key = 'imaginary': imaginary part
        """
        # in general psi_tot will be a complex function
        psi_tot = {'real': np.zeros(len(x_array)), 'imaginary': np.zeros(len(x_array))}

        for n in self.n_coeffs.keys():
            # compute factorial via np.math.factorial()
            if n <= 16:
                a_n = np.float_power(self.mass * self.omega / (np.pi * self.h_bar), 0.25) \
                      / np.sqrt(2**n * np.math.factorial(n))
            # compute factorial via Stirling Approximation
            else:
                a_n = np.float_power(self.mass * self.omega / (np.pi * self.h_bar), 0.25) \
                      / (np.float_power(2 * np.pi * n, 0.25) * np.float_power(2 * n / np.e, n * 0.5))
            hermite_n = special.eval_hermite(n, self.alpha * x_array)
            psi_n = a_n * hermite_n * np.exp((- (self.alpha * x_array)**2 / 2))

            # iteratively add real and imaginary parts of psi_tot
            psi_tot['real'] += complex(self.n_coeffs[n]).real * psi_n
            psi_tot['imaginary'] += complex(self.n_coeffs[n]).imag * psi_n

        return psi_tot

    def eval_probability_distribution(self, x_array: iterable):
        """
        Computes probability distribution,
        based on eigenfunctions psi_n(x) = A_n * H_n(alpha * x) * exp(-alpha^2 * x^2 / 2)
        :parameter x_array: array containing points on the x-position axis
        :returns array containing probability distribution evaluated in x_array
        """
        wave = self.eval_wavefunction(x_array)
        return wave['real'] ** 2 + wave['imaginary'] ** 2


def add(psi: QuantumOscillator, phi: QuantumOscillator, new_name=" "):
    """
    Adds two HarmonicOscillatorEnergyStates with same omega parameter, resulting state is then normalized.
    :parameter psi: first HarmonicOscillatorEnergyState object
    :parameter phi: second HarmonicOscillatorEnergyState object
    :parameter new_name: name of new state, default=' '
    :returns HarmonicOscillatorEnergyState object
    """
    if psi.omega == phi.omega:
        set_psi_keys = set(psi.n_state.keys())
        set_phi_keys = set(phi.n_state.keys())

        # setting up logical sets
        double_keys = set_psi_keys & set_phi_keys
        all_keys = sorted(set_psi_keys | set_phi_keys)  # n_list
        is_in_both = {n: (n in double_keys) for n in all_keys}

        # prepping for coefficients array get
        n_vector = np.identity(len(all_keys))
        psi_vector = {n: psi.n_coeffs[n] * n_vector[i] if n in set_psi_keys
                      else 0 * n_vector[i] for i, n in enumerate(all_keys)}

        phi_vector = {n: phi.n_coeffs[n] * n_vector[i] if n in set_phi_keys
                      else 0 * n_vector[i] for i, n in enumerate(all_keys)}

        # getting coefficients array
        all_coefficients = []
        for n, it_is in is_in_both.items():
            if it_is:
                all_coefficients.append(psi_vector[n] + phi_vector[n])
            elif n in set_psi_keys:
                all_coefficients.append(psi_vector[n])
            elif n in set_phi_keys:
                all_coefficients.append(phi_vector[n])

        all_coefficients = sum(all_coefficients)  # c_list

        return QuantumOscillator(all_keys, all_coefficients,
                                 omega=psi.omega, h_bar=psi.h_bar, name=new_name)

    else:
        print("ERROR: cannot add 2 states with different omegas (not in same Hilbert Space).")
        return None


def modulus_squared(x):
    """
    Computes modulus squared of complex number or array of complex numbers.
    :returns modulus squared (real number)
    """
    z = complex(x)
    z_conj = z.conjugate()
    return (z * z_conj).real


if __name__ == '__main__':

    state_0 = QuantumOscillator([0, 6], (1, np.sqrt(3)), omega=1, name='psi')
    state_1 = QuantumOscillator([3, 9], [np.sqrt(3), 1], omega=1, name='phi')

    print(f'|0> = {state_0.state_vector}, \t{state_0.n_state}\n'
          f'|1> = {state_1.state_vector}, \t\t\t\t\t{state_1.n_state}')

    print(f'N|0> = {round(state_0.number(), 6)}\n'
          f'N|1> = {state_1.number()}')

    a0 = state_0.annihilation()
    a1 = state_1.annihilation()
    print(f'a|0> = {a0}')
    print(f'a|1> = {a1}')

    print(f'|E0> = {state_0.state_vector}, \t{state_0.n_state}\n'
          f'|E1> = {state_1.state_vector}, \t\t\t\t\t{state_1.n_state}')

    aa0 = state_0.annihilation()
    aa1 = state_1.annihilation()
    print(f'a|E0> = {aa0}')
    print(f'a|E1> = {aa1}')

    print(f'|0> = {state_0.state_vector}, \t{state_0.n_state}\n'
          f'|1> = {state_1.state_vector}, \t\t\t\t\t{state_1.n_state}')

    new_state = add(state_0, state_1, new_name='PSI')
    new_state.display()
    print(f'<6|{new_state.name}> = {new_state.n_probability(6)}')

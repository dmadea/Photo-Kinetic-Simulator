
from sympy import Function, solve, Eq, factor, simplify, Symbol, symbols, lambdify
from IPython.display import display, Math
import re

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats.mstats import gmean

from typing import List, Union, Tuple

COLORS = ['blue', 'red', 'green', 'orange', 'black', 'yellow']

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# inspiration from https://stackoverflow.com/questions/4998629/split-string-with-multiple-delimiters-in-python
def split_delimiters(text: str, delimiters: Union[List[str], Tuple[str]]) -> List[tuple]:
    """
    Splits the text with delimiters and returns the list of tuples in which
    first entry is the splitted text and the second is the corresponding delimiter
    used. If there is text after last occurence of a delimiter, empty string will
    be used for last delimiter (second entry of last entry of the list).

    Parameters
    ----------
    text : str
        Input string to separate.
    delimiters : list or tuple 
        Delimiters that will be used to split the text.
    """

    if not text:
        return [('', delimiters[0])]

    pattern = '|'.join(map(re.escape, delimiters))

    # it returns the list in which the every second entry is the original delimiter used
    sp = re.split(f'({pattern})', text)

    entries = sp[::2]
    delimiters = sp[1::2]

    if len(delimiters) < len(entries):
        delimiters.append('')  # append '' as a delimiter if text did not end with delimiter

    return list(zip(entries, delimiters))


def get_matrix(parameters: Union[list, tuple, np.ndarray]) -> tuple:
    """
    Parsers the input parameters and finds all the iterable entries. If there are
    multiple iterable entries in the parameters, all have to have the same length 
    and contain numbers. Function returns the matrix of parameters of shape of 
    k x n where k is the length of iterable entries and n is the number of parameters.
    If the paramateres did not contain any iterables, the resulting  shape of a matrix
    will be 1 x n. Non-iterable entries are repeated in the corresponding columns of a
    return matrix.

    Parameters
    ----------
    parameters : 
        List/tuple of numbers or other iterables.

    Returns
    ----------
    Returns a tuple of resulting matrix and list of indexes of iterables in parameters.
    """

    n_params = len(parameters)
    is_iterable = list(map(np.iterable, parameters))
    #  check whether the length of arrays within the array is the same
    last_length = None
    arr_idxs = list(filter(lambda idx: is_iterable[idx], range(n_params)))
    for i in arr_idxs:
        new_length = len(parameters[i])
        if last_length is not None and last_length != new_length:
            raise ValueError("Array lengths in parameters array does not match.")
        last_length = new_length

    if last_length is None:
        last_length = 1
    param_matrix = np.empty((last_length, n_params), dtype=np.float64)

    # fill the values into the rates matrix
    for i in range(n_params):
        param_matrix[:, i] = np.asarray(parameters[i]) if i in arr_idxs else parameters[i]

    return param_matrix, arr_idxs


class PhotoKineticSymbolicModel:
    """
    Class representing the Symbolic Model. Can parse the text-based model, display the 
    model in LaTex and construct the corresponding differential equations. Steady state
    approximation can be performed for specified compartments. Finally, the model can
    be simulated and the results plotted.

    Attributes
    ----------
    
    elem_reactions: List[dict]
        List of dictionaries of elementary reactions.
    scheme: str
        Text of input scheme.
    symbols: dict
        Sympy symbols and equations that represents the model.
    last_SS_solution: Dict[Dict[dict[str, sympy object]]]
        Dictionaries containing latest solution of a steady state approximation.
    C_tensor : np.ndarray
        The latest result of the simulation. It is in shape of [k, n, t] where
        k is the number of iterables inside parameters to simulate the model for, 
        n is the number of compartments, and t is number of time points
    
    Examples
    ----------
    Simple first order sequential model:

    >>> text_model = '''
    ... A --> B --> C  // k_1 ; k_2
    ... '''
    >>> model = PhotoKineticSymbolicModel.from_text(text_model)
    >>> model.print_model()
    <IPython.core.display.Math object>
    >>> model.pprint_equations()
    Eq(Derivative(c_{A}(t), t), -k_1*c_{A}(t))
    Eq(Derivative(c_{B}(t), t), k_1*c_{A}(t) - k_2*c_{B}(t))
    Eq(Derivative(c_{C}(t), t), k_2*c_{B}(t))
    >>> print(model.symbols['rate_constants'], model.symbols['compartments'])
    [k_1, k_2] [c_{A}(t), c_{B}(t), c_{C}(t)]
    >>> model.simulate_model([1, 0.5], [1, 0, 0], t_max=10, plot_separately=False)

    Excitation and then decompositions of the compound from its singlet state.
    The steady state approximation for the singlet state is performed as the
    lifetime of the state is very short compared to the measured steady state kinetics.

    >>> text_model = '''
    ... GS -hv-> ^1S --> GS  // k_s  # absorption to singlet and decay to ground state
    ... ^1S --> P            // k_r  # formation of product from the singlet state
    ... '''
    >>> model = PhotoKineticSymbolicModel.from_text(text_model)
    >>> model.print_model()
    <IPython.core.display.Math object>
    >>> model.pprint_equations()
    Eq(Derivative(c_{GS}(t), t), -J*(1 - 1/10**(epsilon*l*c_{GS}(t))) + k_s*c_{^1S}(t))
    Eq(Derivative(c_{^1S}(t), t), J*(1 - 1/10**(epsilon*l*c_{GS}(t))) - k_r*c_{^1S}(t) - k_s*c_{^1S}(t))
    Eq(Derivative(c_{P}(t), t), k_r*c_{^1S}(t))
    >>> model.steady_state_approx(['^1S'])
    Eq(Derivative(c_{GS}(t), t), -J*k_r*(10**(epsilon*l*c_{GS}(t)) - 1)/(10**(epsilon*l*c_{GS}(t))*(k_r + k_s)))
    Eq(c_{^1S}(t), J*(10**(epsilon*l*c_{GS}(t)) - 1)/(10**(epsilon*l*c_{GS}(t))*(k_r + k_s)))
    Eq(Derivative(c_{P}(t), t), J*k_r*(10**(epsilon*l*c_{GS}(t)) - 1)/(10**(epsilon*l*c_{GS}(t))*(k_r + k_s)))

    In the following case, we have to specify the incident photon flux and the epsilon at the irradiation wavelength for
    the ground state (GS). Model is simulated for 6 initial concentrations of the ground state from 0.5e-5 to 1.5e-5.
    Note that the initial absorbance of the compound is given by A = l * c * epsilon.

    >>> print(model.symbols['rate_constants'], model.symbols['compartments'])
    [k_s, k_r] [c_{GS}(t), c_{^1S}(t), c_{P}(t)]
    >>> model.simulate_model([1e9, 1e8], [np.linspace(0.5e-5, 1.5e-5, 6), 0, 0], 
    ...                      t_max=500, flux=1e-6, epsilon=1e5, l=1, plot_separately=True)
    """

    delimiters = {
        'absorption': '-hv->',
        'reaction': '-->',
    }

    def __init__(self):
        self.elem_reactions = []  
        self.scheme = ""

        self.symbols = dict(compartments=[],
                            equations=[],
                            rate_constants=[],
                            time=None,
                            flux=None, 
                            l=None,
                            epsilon=None)

        # orders of the rate constants in the model
        self._rate_constant_orders = []
        self.last_SS_solution = dict(diff_eqs={}, SS_eqs={})  # contains dictionaries

        # simulated traces with dimension of k x n x t
        # where k is number of inner parameters to simulate the model for the
        # n is number of compartments and t is number of time points
        self.C_tensor = None 

    @classmethod
    def from_text(cls, scheme: str):
        """
        Takes a text-based model and returns the instance of PhotoKineticSymbolicModel with
        parsed photokinetic model.

        Expected format is single or multiline, forward reactions and absorptions are denoted with 
        '-->' and '-hv->' signs, respecively. Names of species are case sensitive. It is possible to
        denote the sub- or superscirpts with latex notation, e.g. ^1S or H_2O, etc.

        Rate constants for individual reactions can be taken from the text input. They are denoted
        at the end of the each line after '//' characters and are separated by semicolon ';'. If the rate
        constant name is not specified, default name using the reactants and products will be used.
        Comments are denoted by '#'. All characters after this symbol will be ignored.

        Example
        ----------
        Absorption and formation of singlet state, triplet and then photoproducts which
        irreversibly reacts with the ground state
            GS -hv-> S_1 --> GS // k_S  # absorption and singlet state decay

            S_1 --> T_0 --> GS // k_{isc} ; k_T  # intersystem crossing and triplet decay

            T_0 --> P // k_P  # reaction from triplet state to form the products

            P + GS -->  // k_r  # products reacts irreversibly with the ground state

        Parameters
        ----------
        scheme : str
            Imput text-based model.

        Returns
        ----------
        Model representing input reaction scheme.
        """

        if scheme.strip() == '':
            raise ValueError("Parameter scheme is empty!")

        _model = cls()
        _model.scheme = scheme

        # find any number of digits that are at the beginning of any characters
        pattern = re.compile(r'^(\d+).+')

        inv_delimiters = dict(zip(cls.delimiters.values(), cls.delimiters.keys()))

        for line in filter(None, scheme.split('\n')):  # filter removes empty entries from split
            line = line.strip()
            
            line = line.split('#')[0]  # remove comments, take only the characters before possible #

            if line == '':
                continue

            line, *rates = list(filter(None, line.split('//')))  # split to extract rate constant names

            sp_rates = None
            if len(rates) > 0:
                # delimiter for separating rate constants is semicolon ;
                sp_rates = list(map(str.strip, filter(None, rates[0].split(';'))))

            tokens = []

            delimiters = list(cls.delimiters.values())
            for side, delimiter in filter(lambda e: e[0] != '',  split_delimiters(line, delimiters)):  # remove empty entries
                entries = []

                # process possible number in front of species, species cannot have numbers in their text
                for entry in filter(None, side.split('+')):
                    entry = ''.join(filter(lambda d: not d.isspace(), list(entry)))  # remove white space chars

                    if entry == '':
                        continue

                    match = re.match(pattern, entry)

                    number = 1
                    if match is not None:  # number in front of a token
                        str_num = match.group(1)   #  get the number in front of the entry name
                        entry = entry[len(str_num):]
                        number = int(str_num)

                    if number < 1:
                        number = 1

                    entries += number * [entry]  # list arithmetics

                tokens.append([entries, delimiter])

            num_of_reactions = len(list(filter(lambda token: token[1] == cls.delimiters['reaction'], tokens)))

            if sp_rates and len(sp_rates) > 0:
                # if the numbers of reactions and rate constants does not match, lets use just the first entry
                if len(sp_rates) != num_of_reactions:
                    sp_rates = num_of_reactions * [sp_rates[0]]
            else:
                sp_rates = num_of_reactions * [None]

            rate_iterator = iter(sp_rates)
            for i in range(len(tokens)):
                rs, r_del = tokens[i]
                ps, _ = tokens[i + 1] if i < len(tokens) - 1 else ([], None)

                if r_del == '':
                    continue

                r_type = inv_delimiters[r_del]
                r_name = next(rate_iterator) if r_type == 'reaction' else 'h\\nu'

                if r_name is None:
                    r_name = f"k_{{{'+'.join(rs)}{'+'.join(ps)}}}"
                    # print(r_name)

                if r_type == 'absorption' and len(ps) == 0:
                    raise ValueError(f"Missing species after absorption arrow ({cls.delimiters['absorption']}).")

                _model.add_elementary_reaction(rs, ps, type=r_type, rate_constant_name=r_name)

        _model._build_equations()

        return _model

    def add_elementary_reaction(self, from_comp: Union[List[str], Tuple[str]] = ('A', 'A'),
                                      to_comp: Union[List[str], Tuple[str]] = ('B', 'C'),
                                      type: str = 'reaction',
                                      rate_constant_name: Union[None, str] = None):
        """
        Adds the elementary reaction to the model.

        Parameters
        ----------
        from_comp : 
            List/tuple of reactants. Default ('A', 'A').
        to_comp : 
            List/tuple of products. Can be empty list/tuple. Default ('B', 'C')
        type: 
            Type of the elementary reaction. Can be either 'reaction' or 'absorption'.
        rate_constant_name: 
            Name of the rate constant. Optional.
        """

        from_comp = from_comp if isinstance(from_comp, (list, tuple)) else [from_comp]
        to_comp = to_comp if isinstance(to_comp, (list, tuple)) else [to_comp]

        el = dict(from_comp=from_comp, to_comp=to_comp, type=type, rate_constant_name=rate_constant_name)

        if el in self.elem_reactions:
            return

        self.elem_reactions.append(el)

    def print_model(self):
        """
        Print the model in LaTeX format. Chooses the correct way of printing the model based 
        on the environment (Jupyter notebook or Google colab).
        """

        if IN_COLAB:
            self.pprint_model_no_env()
        else:
            self.pprint_model_align_env()

    def pprint_model_align_env(self):
        """Pretty prints model. Uses the align LaTeX environment."""

        latex_eq = ''

        for el in self.elem_reactions:
            reactants = ' + '.join([f'\\mathrm{{{comp}}}' for comp in el['from_comp']])
            products = ' + '.join([f'\\mathrm{{{comp}}}' for comp in el['to_comp']])

            r_name = el['rate_constant_name']

            latex_eq += f'{reactants} &\\xrightarrow{{{r_name}}} {products} \\\\'

        latex_eq = r'\begin{align}' + latex_eq + r'\end{align}'
        display(Math(latex_eq))

    def pprint_model_no_env(self):
        """Pretty prints model. Does not use the math environment."""

        for el in self.elem_reactions:
            reactants = ' + '.join([f'\\mathrm{{{comp}}}' for comp in el['from_comp']])
            products = ' + '.join([f'\\mathrm{{{comp}}}' for comp in el['to_comp']])

            r_name = el['rate_constant_name']

            latex_eq = f'{reactants} \\xrightarrow{{{r_name}}} {products}'

            display(Math(latex_eq))

    def pprint_equations(self, subs: List[tuple] = None):
        """
        Pretty prints the equations. It allows to substitute some parts of the equation
        with desired symbols.
        
        Parameters
        ----------
        subs : 
            List of tuples. In each tuple, first entry is the old expression and second
            entry is the new expression. Default is None.
        """

        if self.symbols['equations'] is None:
            return

        for eq in self.symbols['equations']:
            _eq = eq
            if subs:
                for old, new in subs:
                    _eq = _eq.subs(old, new)

            display(_eq)

    def get_compartments(self) -> list:
        """
        Return the compartment names in the model, the names are case sensitive.
        """
        names = []
        for el in self.elem_reactions:
            for c in el['from_comp']:
                if c not in names:
                    names.append(c)

            for c in el['to_comp']:
                if c not in names:
                    names.append(c)
        return names

    def steady_state_approx(self, compartments: Union[List[str], Tuple[str]],
                            print_solution: bool = True):
        """
        Performs the steady state approximation for the given species.

        Parameters
        ----------
        compartments : 
            List/tuple of compartments given as strings for which steady state
            approximation will be perfomerd. Names of compartments must be as
            those used for constructing the model.
        print_solution: 
            If True, the solution will be pretty printed. Default True.
        """

        if self.symbols['equations'] is None:
            return

        self.last_SS_solution['diff_eqs'].clear()
        self.last_SS_solution['SS_eqs'].clear()

        eq2solve = []   
        variables = []

        all_compartments = self.get_compartments()

        for comp, eq, f in zip(all_compartments, self.symbols['equations'], self.symbols['compartments']):
            if comp in compartments:  # use SS approximation
                eq2solve.append(Eq(eq.rhs, 0))
                variables.append(f)
            else:  # keep the diff equation
                eq2solve.append(eq) 
                variables.append(f.diff(self.symbols['time']))

        solution = solve(eq2solve, variables)

        if len(solution) == 0:
            raise ValueError('Steady state solution for the given input does not exist.')

        # the order of solutions is the same as the input
        for comp, (var, expression) in zip(all_compartments, solution.items()):
            eq = Eq(var, expression)
            eq = factor(simplify(eq))

            # TODO -> allow substitutions
            # if subs:
            #     for old, new in subs:
            #         eq = eq.subs(old, new)

            if comp in compartments:
                self.last_SS_solution['SS_eqs'][comp] = eq
            else:
                self.last_SS_solution['diff_eqs'][comp] = eq
            if print_solution:
                display(eq)

    def clear_model(self):
        """Clears the model."""

        self.symbols['compartments'].clear()
        self.symbols['equations'].clear()
        self.symbols['rate_constants'].clear()
        self.symbols['time'] = None
        self.symbols['flux'] = None
        self.symbols['l'] = None
        self.symbols['epsilon'] = None
        self._rate_constant_orders.clear()
        self.C_tensor = None
        self.scheme = ''


    def _build_equations(self):
        """
        Converts the elementary reactions to sympy representation of differential equations.
        """

        self.clear_model()

        comps = self.get_compartments()

        # right hand sides of diff. equations
        sympy_rhss = len(comps) * [0]

        # time and concentrations of absorbed photons J
        self.symbols['time'], self.symbols['flux'] = symbols('t J')
        self.symbols['l'], self.symbols['epsilon'] = symbols('l epsilon')

        for c in comps:
            # f = Function(f'[{{{c}}}]')(s_t)
            f = Function(f'c_{{{c}}}')(self.symbols['time'])
            self.symbols['compartments'].append(f)

        idx_dict = dict(enumerate(comps))
        inv_idx = dict(zip(idx_dict.values(), idx_dict.keys()))

        r_names = []
        for el in self.elem_reactions:
            if el['type'] != 'reaction':
                continue

            r_names.append(el['rate_constant_name'])
            self._rate_constant_orders.append(len(el['from_comp']))

        self.symbols['rate_constants'] = list(map(Symbol, r_names))

        # symbolic rate constants dictionary
        s_rates_dict = dict(zip(r_names, self.symbols['rate_constants']))

        for el in self.elem_reactions:
            i_from = list(map(lambda com: inv_idx[com], el['from_comp']))  # list of indexes of starting materials
            i_to = list(map(lambda com: inv_idx[com], el['to_comp']))  # list of indexes of reaction products

            if el['type'] == 'absorption':

                k = i_from[0]  # absorption for more compartments does not make sense
                # 1 - 10 ** (-l * eps * c)
                fraction_abs = 1 - 10 ** (-self.symbols['l'] * self.symbols['epsilon'] * self.symbols['compartments'][k])

                sympy_rhss[k] -= self.symbols['flux'] * fraction_abs

                k = i_to[0]
                sympy_rhss[k] += self.symbols['flux'] * fraction_abs

                continue

            forward_prod = s_rates_dict[el['rate_constant_name']]

            for k in i_from:
                forward_prod *= self.symbols['compartments'][k]  # forward rate products, eg. k_AB * [A] * [B]

            for k in i_from:
                sympy_rhss[k] -= forward_prod   # reactants

            for k in i_to:
                sympy_rhss[k] += forward_prod   # products

        for f, rhs in zip(self.symbols['compartments'], sympy_rhss):
            # construct differential equation
            _eq = Eq(f.diff(self.symbols['time']), rhs)
            self.symbols['equations'].append(_eq)

    def simulate_model(self, rate_constants: Union[List, Tuple, np.ndarray],
                       initial_concentrations: Union[List, Tuple, np.ndarray],
                       constant_compartments: Union[None, List[str], Tuple[str]] = None, 
                       t_max: Union[float, int] = 1e3, t_points: int = 1000, flux: float = 1e-8, l: float = 1,
                       epsilon: float = 1e5, use_SS_approx: bool = True, ODE_solver: str = 'Radau', 
                       plot_separately: bool = True,  figsize: Union[tuple, list] = (6, 3), yscale: str = 'linear',
                       cmap: str = 'plasma', lw: float = 2, legend_fontsize: int = 10, legend_labelspacing: float = 0,
                       filepath: Union[None, str] = None, dpi: int = 300, transparent: bool = False,
                       plot_results: bool = True, scale: bool = True):
        """
        Simulates the current model and plots the results if ``plot_results`` is True (default True). Parameters
        rate_constant and initial_concentrations can contain iterables. In this case, the model will be simulated
        for each of the value in the iterable arrays. Multiple iterables are allowed. In such case, lengths of
        iterable arrays in parameters rate_constants and intial_concentrations has to be equal. The result of
        the simulation is saved in a class attribute C_tensor.

        Parameters
        ----------
        rate_constants : 
            List/tuple of rate constants. They have to occur in the same order as are saved in the model.
            Please, check the order in ``symbols['rate_constants']`` attribute. Rate constants can contain
            iterables (list, tuple). In this case the model will be simulated for each of the parameter in
            the iterable. It can contain more that one iterables, in this case, length of iterable arrays must 
            be the same.
        initial_concentrations : 
            List/tuple of initial concentrations. They have to occur in the same order as are saved in the model.
            Please, check the order in ``symbols['compartments']`` attribute. Initial concentrations can contain
            iterables (list, tuple). In this case the model will be simulated for each of the parameter in
            the iterable. It can contain more that one iterables, in this case, length of iterable arrays must 
            be the same.
        constant_compartments: list of strings
            If specified (default None), the values of these compartments will be constant and the value from
            initial_concentration array will be used. Constant compartments will not be plotted. If the SS 
            approximation was performed for the constant compartment, it will have not effect and the SS 
            solution will be used instead.
        t_max: 
            Maximum time in the range of time points the model will be simulated for. Optional. Default 1e3.
        t_points: 
            Number of time points used to simulate the model for. Optional. Default 1000.
        flux: 
            Incident photon flux. Optional. Default 1e-8.
        l: 
            Length of the cuvette. Optional. Default 1.
        epsilon: 
            Molar absorption coefficient at the irradiation wavelength. Optional. Default 1e5.
        use_SS_approx: 
            If True (default True), the equations from SS approximation will be used to simulate the model
            (only if it was performed for the model).
        ODE_solver: 
            Name of the ODE solver used to numerically simulate the model. Default 'Radau'. Available options are:
            'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'. For details, see
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp

        plot_separately: 
            If True (default True), the plot will have n graphs in which n is number of compartments to plot. In each graph,
            the one compartment will be plotted. If there are some iterables (length k) in the input parameters, each graph
            will have k curves.
        figsize: tuple 
            Figure size of one graph. Optional. Default (6, 3): (width, height).
        yscale: 
            Scale of the y axis. Optional. Default 'linear'. Could be 'log', etc.
        cmap: 
            Color map used to color the individual curves if plot_separately = True.
        lw: 
            Line width of the curves. Optional. Default 2.
        legend_fontsize: 
            Fontsize of the legend. Optional. Default 10.
        legend_labelspacing: 
            Vertical spacing of the labels in legend. Optional. Default 0. Can be negative.
        filepath: 
            If specified (default None), the plot will be saved into this location with a specified filename.
        dpi: 
            DPI of the resulting plot that will be saved to file. Optional. Default 300.
        transparent: 
            If True, the saved image will be transparent. Optional. Default False.
        plot_results: 
            If True, the result will be plotted. Optional. Default True.
        scale: 
            If True, rate constant, initial concentrations and epsilon and flux will be scaled by suitable factor
            (determined by the geometric mean of the initial concentrations). This can help to reduce numerical errors
            in the integrator. Optional. Default True.
        """
        
        if len(self.symbols['equations']) == 0:
            raise ValueError("No model instantiated.")

        comps = self.get_compartments()
        n = len(comps)
        use_SS_approx = use_SS_approx and len(self.last_SS_solution['SS_eqs'].values()) > 0

        # find the number of arrays in the rate_constant array
        n_rates = len(self.symbols['rate_constants'])

        assert len(initial_concentrations) == n, "Length of initial_concentrations must match the number of compartments."
        assert len(rate_constants) == n_rates, "Length of rate_constants must match the number of rate constants."

        # shape of matrices are k x l where k is number of parameters within the array to simulated
        # and l number of parameters (n_rates or n)
        rates, idx_iter_rates = get_matrix(rate_constants)
        init_c, idx_iter_init_c = get_matrix(initial_concentrations)

        k = max(rates.shape[0], init_c.shape[0])  # number of inner parameters in a both parameter arrays
        
        # tile the other array so the first dimension of both arrays is k
        if rates.shape[0] < init_c.shape[0]:
            rates = np.tile(rates[0], (k, 1))
        elif rates.shape[0] > init_c.shape[0]:
            init_c = np.tile(init_c[0], (k, 1))

        self.C_tensor = None
        times = np.linspace(0, t_max, int(t_points))

        # scale rate constants, flux, epsilon and initial_concentrations for less numerial errors in
        # the numerical integration
        coef = 1
        init_c_sc = init_c.copy()
        rates_sc = rates.copy()
        if scale:
            # scaling coefficient, calculate it as geometric mean from non-zero initial concentrations
            coef = gmean(init_c[init_c > 0])  
            # print(coef)
            init_c_sc /= coef
            epsilon *= coef
            flux /= coef
            # second and higher orders needs to be appropriately scaled, by coef ^ (rate order - 1)
            rates_sc *= coef ** (np.asarray(self._rate_constant_orders, dtype=np.float64) - 1)

        symbols = self.symbols['rate_constants'] + self.symbols['compartments'] + [self.symbols['flux'], self.symbols['l'], self.symbols['epsilon']]

        sym_eqs = list(self.last_SS_solution['diff_eqs'].values()) if use_SS_approx else self.symbols['equations']

        d_funcs = list(map(lambda eq: lambdify(symbols, eq.rhs, 'numpy'), sym_eqs))

        # those compartments that will be simulated numerically
        idxs2simulate = list(map(comps.index, self.last_SS_solution['diff_eqs'].keys())) if use_SS_approx else list(np.arange(n, dtype=int))

        idxs_constant = []  # empty list
        idxs_constant_cast = []
        if constant_compartments is not None:
            idxs_constant = map(comps.index, filter(lambda c: c in comps, constant_compartments))
            idxs_constant_cast = list(map(idxs2simulate.index, filter(lambda c: c in idxs2simulate, idxs_constant)))

        def dc_dt(t, c, rates, flux, l, epsilon):
            _c = np.empty(n)  # need to cast the concentations to have the original size of the compartments
            _c[idxs2simulate] = c

            solution = np.asarray([f(*rates, *_c, flux, l, epsilon) for f in d_funcs])
            solution[idxs_constant_cast] = 0  # set the change of constant compartments to zero

            return solution

        C = np.empty((k, n, times.shape[0]))

        for i in range(k):
            # numerically integrate
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
            sol = solve_ivp(dc_dt, (0, t_max), init_c_sc[i, idxs2simulate], method=ODE_solver, vectorized=False, dense_output=True,
                    args=(rates_sc[i, :], flux, l, epsilon))

            C[i, idxs2simulate, :] = sol.sol(times)  # evaluate at dense time points

            # calculate the SS concentrations from the solution of diff. eq.
            if use_SS_approx:
                for comp, eq in self.last_SS_solution['SS_eqs'].items():
                    j = comps.index(comp)
                    f = lambdify(symbols, eq.rhs, 'numpy')
                    C[i, j, :] = f(*rates_sc[i, :], *list(C[i]), flux, l, epsilon)

        # scale the traces back to correct values
        C *= coef
        self.C_tensor = C

        # create indexes of only those comps. which will be plotted, so remove all constant comps.
        idxs2plot = set(np.arange(n, dtype=int)) - set(np.asarray(idxs2simulate)[idxs_constant_cast])

        comps_cast = [comps[i] for i in idxs2plot]
        traces_cast = [self.C_tensor[:, i, :] for i in idxs2plot]

        # find what rate constants or initial concentrations are changing
        par_names = []
        for i in range(k):
            # https://docs.python.org/3/library/string.html#format-string-syntax
            # # option does not remove the trailing zeros from the output
            text_rates = ', '.join([f"{self.symbols['rate_constants'][j]}={rates[i, j]:#.3g}" for j in idx_iter_rates])
            text_init = ', '.join([f"[\\mathrm{{{comps[j]}}}]_0={init_c[i, j]:#.3g}" for j in idx_iter_init_c])

            # combine texts and remove empty entries (in case the the texts are empty)
            par_names.append('; '.join(filter(None, [text_rates, text_init])))

        if not plot_results:
            return

        # plot the results
        if plot_separately:
            n_rows = n - len(idxs_constant_cast)
            figsize = (figsize[0] , figsize[1] * n_rows)
            fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)

            # colormap for inner parameters
            cmap = cm.get_cmap(cmap)
            for i, (comp, trace, ax) in enumerate(zip(comps_cast, traces_cast, axes)):
                for j in range(k):
                    label = '' if par_names[j] == '' else f'${par_names[j]}$'
                    ax.plot(times, trace[j], label=label if i == 0 else '', lw=lw, color=cmap(j/trace.shape[0]))
                ax.set_ylabel('Concentration')
                if i == 0 and k > 1:
                    ax.legend(frameon=False, fontsize=legend_fontsize, labelspacing=legend_labelspacing)
                ax.set_yscale(yscale)
                ax.set_title(f'$\\mathrm{{{comp}}}$')
                ax.tick_params(axis='both', which='major', direction='in')
                ax.tick_params(axis='both', which='minor', direction='in')
                ax.xaxis.set_ticks_position('both')
                ax.yaxis.set_ticks_position('both')
            axes[-1].set_xlabel('Time')

        else:
            figsize = (figsize[0] , figsize[1] * k)
            fig, axes = plt.subplots(k, 1, figsize=figsize, sharex=True)

            for i in range(k):
                ax = axes[i] if k > 1 else axes

                for j, (comp, trace) in enumerate(zip(comps_cast, traces_cast)):
                    color = COLORS[j % len(COLORS)]
                    ax.plot(times, trace[i], label=f'$\\mathrm{{{comp}}}$', lw=lw, color=color)

                if i == k - 1:
                    ax.set_xlabel('Time')
                ax.set_ylabel('Concentration')
                title = '' if par_names[i] == '' else f'${par_names[i]}$'
                ax.set_title(title)
                ax.set_yscale(yscale)
                ax.tick_params(axis='both', which='major', direction='in')
                ax.tick_params(axis='both', which='minor', direction='in')
                ax.xaxis.set_ticks_position('both')
                ax.yaxis.set_ticks_position('both')
                ax.legend(frameon=False, fontsize=legend_fontsize, labelspacing=legend_labelspacing)
            
        plt.tight_layout()
        if filepath:
            plt.savefig(filepath, dpi=dpi, transparent=transparent)
        plt.show()

    def print_text_model(self):
        """Print the model as text."""
        print(f'Scheme: {self.scheme}')

        for el in self.elem_reactions:
            rate = el['rate_constant_name']# if el['type'] == 'reaction' else None
            print(f"{el['type'].capitalize()}: {' + '.join(el['from_comp'])} \u2192 {' + '.join(el['to_comp'])}, "
                  f"rate: {rate}")  # {rate=} does not work in python 3.7 on colab


if __name__ == '__main__':

    model = """
    ArO_2 --> Ar + ^1O_2             // k_1  # absorption and singlet state decay
    ^1O_2 --> ^3O_2                  // k_d
    # ^1O_2 + Ar --> Ar + ^3O_2         // k_{q,Ar}
    # ^1O_2 + ArO_2 --> ArO_2 + ^3O_2     // k_{q,ArO_2}
    ^1O_2 + S --> S + ^3O_2           // k_{q,S}
    S + ^1O_2 -->                      // k_r
    """


    # model = PhotoKineticSymbolicModel.from_text(model)
    # # print(model.print_text_model())

    # # model.simulate_model([1, 10], [0.1, 0], t_max=10, yscale='linear', scale=False)


    # model.steady_state_approx(['^1O_2'], print_solution=False)
    # model.simulate_model([np.linspace(1e-3, 5e-3, 6), 1/9.5e-6, 1e4, 1e9], [1e-3, 0, 0, 95, 1e-3],
    #                     constant_compartments=['^3O_2'], t_max=2e3, yscale='linear', scale=True,
    #                     plot_separately=False, cmap='plasma')


    text_model = """
    A --> B --> C  // k_1 ; k_2
    """

    model = PhotoKineticSymbolicModel.from_text(text_model)
    model.print_model()
    print(model.symbols['rate_constants'], model.symbols['compartments'])
    # model.simulate_model([1, 0.5], [1, 0, 0], t_max=10, plot_separately=False)
    model.pprint_equations()
    # model.simulate_model([1e9, 1e8], [np.linspace(0.5e-5, 1.5e-5, 6), 0, 0], t_max=500, flux=1e-6, epsilon=1e5, plot_separately=True)








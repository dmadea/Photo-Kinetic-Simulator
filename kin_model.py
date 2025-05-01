from sympy import Function, solve, Eq, factor, simplify, Symbol, symbols, lambdify, pprint, init_printing, Add, pi, sqrt, exp, ln, Piecewise
from IPython.display import display, Math, Markdown
import re

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats.mstats import gmean

from numba import vectorize

from typing import Callable, Iterable, List, Union, Tuple

COLORS = ['blue', 'red', 'green', 'orange', 'black', 'yellow']

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'xtick.major.size': 5, 'ytick.major.size': 5})
plt.rcParams.update({'xtick.minor.size': 2.5, 'ytick.minor.size': 2.5})
plt.rcParams.update({'xtick.major.width': 1, 'ytick.major.width': 1})
plt.rcParams.update({'xtick.minor.width': 0.8, 'ytick.minor.width': 0.8})

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


def get_matrix(parameters: Union[None, list, tuple, np.ndarray]) -> tuple:
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

    if parameters is None:
        return np.empty((1, 0), dtype=np.float64), []

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


def format_number_latex(number: float, sig_figures: int = 3) -> str:
    """
    Formats the number in latex format and round it to defined significant figures.
    If the result is in the exponential format, it will be formatted as
    ``[number] \\times 10^{[exponent]}``.

    Parameters
    ----------
    number :
        Number to format.
    sig_figures:
        Number of significant figures. Optional. Default 3.
    """

    formatted_num = f'{number:#.{sig_figures}g}'

    if 'e' in formatted_num:
        num_str, exponent = formatted_num.split('e')

        return f'{num_str} \\times 10^{{{int(exponent)}}}'

    return formatted_num

@vectorize(nopython=True, fastmath=False)
def photokin_factor(A_prime: float, l: float) -> float:
    """
    Returns the photokinetic factor for a given A_prime and l.

    Parameters
    ----------
    A_prime : float
        Product of the concentration and the epsilons, A_prime = sum(eps * c)
    l : float
        The path length.
    """
    ln10 = np.log(10)
    ll2 = (l * ln10) ** 2 / 2

    A = A_prime * l

    if A < 1e-3:
        return l * ln10 - A * ll2  # approximation with first two taylor series terms
    else:
        return (1 - np.exp(-A * ln10)) / A_prime  # exact photokinetic factor

def gaussian(times: np.ndarray | float, FWHM: float, J0: float = 1) -> np.ndarray | float:
    """
    Returns the value of the Gaussian pulse at a given time.
    """

    ln2 = np.log(2)
    return J0 * 2 * np.sqrt(ln2) * np.exp(- 4 * ln2 * times ** 2 / FWHM ** 2) / (FWHM * np.sqrt(np.pi))


def square(times: np.ndarray | float, sw: float = 1, J0: float = 1) -> np.ndarray | float:
    """
    Returns the value of the square pulse at a given time.
    """
    return J0 * ((times >= 0) & (times <= sw))


def get_time_unit(time: float) -> Tuple[float, str]:
    """
    Finds the nearest unit scale and returns the scaling factor
    and the unit.

    Parameters
    ----------
    time :
        Time to format.

    Returns
    ----------
    Returns a tuple of scaling factor and time unit. By multipling the 
    original time with the scaling factor, time in the determined unit scale 
    will be obtained.
    """
    f, p, n, u, m = np.logspace(-15, -3, 5)

    unit = 's'
    scaling_factor = 1
    if f <= time < p:
        scaling_factor = f 
        unit = 'fs'
    elif p <= time < n:
        scaling_factor = p
        unit = 'ps'
    elif n <= time < u:
        scaling_factor = n
        unit = 'ns'
    elif u <= time < m:
        scaling_factor = u
        unit = '\\mu s'
    elif m <= time < 1:
        scaling_factor = m
        unit = 'ms'
    elif 60 <= time < 3600:
        scaling_factor = 60
        unit = 'min'
    elif time >= 3600:
        scaling_factor = 3600
        unit = 'h'

    return 1 / scaling_factor, unit


def format_time_latex(number: float, sig_figures: int = 3):
    tol = 1-10**(-sig_figures) + 5 * 10 ** (-sig_figures - 1)
    f, p, n, u, m = np.logspace(-15, -3, 5) * tol

    unit = 's'
    if f <= number < p:
        number *= 1e15 
        unit =  'fs'
    elif p <= number < n:
        number *=  1e12
        unit =  'ps'
    elif n <= number < u:
        number *=  1e9
        unit =  'ns'
    elif u <= number < m:
        number *=  1e6
        unit =  '\\mu s'
    elif m <= number < tol:
        number *=  1e3
        unit =  'ms'
    elif tol * 60 <= number < tol * 3600:
        number /=  60
        unit =  'min'
    elif number >= tol * 3600:
        number /=  3600
        unit =  'h'

    formatted_num = f'{number:.{sig_figures}g}'

    if 'e' in formatted_num:
        num, exponent = formatted_num.split('e')
        return f'{num} \\times 10^{{{int(exponent)}}} {unit}'

    return f'{formatted_num}\\ {unit}'


def ode_integrate(func: Callable, y0: np.ndarray | float, times: np.ndarray, method: str = 'Radau', 
                args: tuple | None = None, pulse_max_step: float = np.inf, pulse_duration: float | None = None,
                atol: float = 1e-6, rtol: float = 1e-3, default_max_step: float = np.inf):
    """
    Integrates the ODE system using the specified method.

    Parameters
    ----------
    func : Callable
        The function to integrate.
    y0 : np.ndarray | float
        The initial condition.
    times : np.ndarray
        The times to integrate the ODE system.
    method : str
        Name of the ODE solver used to numerically simulate the model. Default 'Radau'. Available options are:
        'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'.
    args : tuple | None
        The arguments to pass to the function.
    pulse_max_step : float | None
        The maximum step size in the integration of the region of pulse.
    pulse_duration : float | None
        The duration of the pulse. Starts from the beginning.
    atol : float | None
        The absolute tolerance for the integration.
    rtol : float | None
        The relative tolerance for the integration.
    default_max_step : float
        The default maximum step size for the integration. Used when the pulse duration is not specified 
        and the integration is done over the entire time range.
    """

    if pulse_duration is None:
        sol = solve_ivp(func, (times[0], times[-1]), y0, method=method, vectorized=False, dense_output=False,
            t_eval=times, max_step=default_max_step, args=args, atol=atol, rtol=rtol)

        return sol.y

    else:
        times_pulse = times[times <= times[0] + pulse_duration]
        times_rest = times[times_pulse[-1] <= times]

        # times_pulse[-1] == times_rest[0]

        sol_pulse = solve_ivp(func, (times[0], times_pulse[-1]), y0, method=method, vectorized=False, dense_output=False,
            t_eval=times_pulse, max_step=pulse_max_step, args=args, atol=atol, rtol=rtol)

        sol_rest = solve_ivp(func, (times_rest[0], times[-1]), sol_pulse.y[:, -1], method=method, vectorized=False, dense_output=False,
            t_eval=times_rest, max_step=default_max_step, args=args, atol=atol, rtol=rtol, first_step=pulse_max_step)

        return np.hstack((sol_pulse.y[:, :-1], sol_rest.y))



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
    concentration_unit : 
        The unit of concentration that will be used to denote the graph axes and 
        will be used for units of rate constants (those that have an order higher 
        than first order).
    
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

    Flux_types = ['Gaussian pulse', 'Square pulse', 'Continuous']

    def __init__(self, concentration_unit: str = 'M', flux_type: str = 'Continuous'):
        self.elem_reactions = []  
        self.scheme = ""

        self.flux_type: str = flux_type
        self.flux_equations: list[Eq] = []
        self.explicit_Fk_equation: Eq | None = None

        self.symbols = dict(compartments=[],
                            equations=[],  # contains diff equations in full form
                            equations_Fk=[], # contains diff equations with symbol Fk instead of (1 - 10 ** -A) / A
                            rate_constants=[],
                            time=None,  # t
                            flux=None,    # J(t)
                            Fk=None,  # Photokinetic factor
                            explicit_Fk=None,  # explicit full Fk = (1 - 10 ** -A) / A
                            l=None,
                            epsilons=[],
                            substitutions=[],
                            other_symbols=[])  # contains the symbols used for substitutions

        # orders of the rate constants in the model
        self._rate_constant_orders: list[int] = []
        self.last_SS_solution: dict[str, dict[str, Eq]] = dict(diff_eqs={}, SS_eqs={})  # contains dictionaries
        self.absorbing_compartments: dict[str, int] = {}  # contains the name indicies pairs of absorbing compartments in self.get_compartments() list

        self.C_tensor = None  # concentration tensor
        self.times = None  # times unscaled
        self.A_tensor = None  # absorbance tensor
        self.concentration_unit = concentration_unit
        self.last_parameter_matrix: np.ndarray | None = None

        self.J: Callable | None = None  # flux function, takes time and index as arguments
        self.compartments_to_plot: list[str] = []  # compartments to plot
        self.last_parameter_map: dict[str, dict] = {}  # contains the last symbols map


    def create_flux_equations(self):
        # Flux_types = ['Gaussian pulse', 'Square pulse', 'Continuous']

        sigma, t, sw, J0 = symbols("\\sigma t s_w J_0")
        FWHM = symbols("FWHM")

        self.symbols['other_symbols'] = [sw, J0, FWHM]

        g = J0 * exp( - self.symbols['time'] ** 2 / (2 * sigma ** 2)) / (sigma * sqrt(2 * pi)) 
        g = g.subs(sigma, FWHM / (2 * sqrt(2 * ln(2)))) # substitute sigma with FWHM

        eq = Eq(self.symbols['flux'], g)

        self.flux_equations.append(eq)

        sq = Piecewise(
            (0, t < 0),
            (J0, (t >= 0) & (t <= sw)),
            (0, True)
        )

        eq = Eq(self.symbols['flux'], sq)
        self.flux_equations.append(eq)

        eq = Eq(self.symbols['flux'], J0)
        self.flux_equations.append(eq)

    @property
    def flux_type(self):
        return self._flux_type

    @flux_type.setter
    def flux_type(self, value):
        if value not in self.Flux_types:
            raise ValueError(f"Invalid flux type: {value}. Must be one of {self.Flux_types}")
        self._flux_type = value

    @classmethod
    def from_text(cls, scheme: str, **kwargs):
        """
        Takes a text-based model and returns the instance of PhotoKineticSymbolicModel with
        parsed photokinetic model.

        Expected format is single or multiline, forward reactions and absorptions are denoted with 
        '-->' and '-hv->' signs, respecively. Names of species are case sensitive. It is possible to
        denote the sub- or superscirpts with latex notation, e.g. ^1S or H_2O, etc.

        To denote the species that absorbs light but behaves as inner filter effect, use the following format:
        A -hv-> A    # absorbing species that absorbs light but yield not photoproduct

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
            Input text-based model.

        **kwargs : 
            Other keyword arguments passed to the class constructor.

        Returns
        ----------
        Model representing input reaction scheme.
        """

        if scheme.strip() == '':
            raise ValueError("Parameter scheme is empty!")

        _model = cls(**kwargs)
        _model.scheme = scheme

        # find any number of digits that are at the beginning of any characters
        p_digits = re.compile(r'^(\d+).+')
        # http://www.rexegg.com/regex-best-trick.html
        # we want to find only those + signs not enclosed in curly braces, not after ^ and _ characters
        # therefore, we will take only group 4
        # this cannot handle nested braces..., TODO: fix this
        p_plus_signs = re.compile(r'{[^{}]*(\+)[^{}]*}|\^(\+)|_(\+)|(\+)')  

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
                # gets matches of the found + signs in the side
                matches = re.finditer(p_plus_signs, side)

                # get only spans for group 4
                plus_sign_spans = [m.span(4) for m in matches if m.group(4) is not None]
                plus_sign_spans = [(None, 0)] + plus_sign_spans + [(None, None)]  # add start and end tuples

                splitted_side = []  # split the side based on the spans
                for (s, e), (s1, e1) in zip(plus_sign_spans[:-1], plus_sign_spans[1:]):
                    splitted_side.append(side[e:s1])

                entries = []

                # process possible number in front of species, species cannot have numbers in their text
                for entry in splitted_side:
                    entry = ''.join(filter(lambda d: not d.isspace(), list(entry)))  # remove white space chars

                    if entry == '':
                        continue

                    match = re.match(p_digits, entry)

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

    def print_model(self, force_use_environment: bool = False ):
        """
        Print the model in LaTeX format. If not in Colab, uses the align* LaTeX environment.

        Parameters
        ----------
        force_use_environment : 
            If True, forces to use the align* LaTeX environment for printing the model. Default False.
        """
        # trick with Markdown https://stackoverflow.com/questions/48422762/is-it-possible-to-show-print-output-as-latex-in-jupyter-notebook

        force_use_environment = not IN_COLAB or force_use_environment

        idxs = []  # indexes of reversible reactions
        eqs = []

        and_symbol = '&' if force_use_environment else ''
        for i, el in enumerate(self.elem_reactions):
            if i in idxs:
                continue

            reactants = ' + '.join(map(lambda comp: r'\mathrm{%s}' % comp, el['from_comp']))
            products = ' + '.join(map(lambda comp: r'\mathrm{%s}' % comp, el['to_comp']))

            idx_rev = None
            for j in range(i + 1, len(self.elem_reactions)):
                if el['from_comp'] == self.elem_reactions[j]['to_comp'] and \
                   el['to_comp'] == self.elem_reactions[j]['from_comp'] and \
                   el['type'] == self.elem_reactions[j]['type']:
                    # reversible reaction
                    idx_rev = j
                    idxs.append(j)

            forward_rate = el['rate_constant_name']

            if idx_rev:
                # \xrightleftharpoons does not work in Colab and Jupyter :(
                # eqs.append(f"{reactants} &\\xrightleftharpoons[{self.elem_reactions[idx_rev]['rate_constant_name']}]{{\\hspace{{0.1cm}}{r_name}}}\\hspace{{0.1cm}} {products}")
                backward_rate = self.elem_reactions[idx_rev]['rate_constant_name']
                eqs.append(r"%s\ %s\overset{%s}{\underset{%s}\rightleftharpoons}\ %s" % (reactants, and_symbol, forward_rate, backward_rate, products))
            else:
                eqs.append(r"%s %s\xrightarrow{%s} %s" % (reactants, and_symbol, forward_rate, products))

            if not force_use_environment:
                display(Math(eqs[-1]))
            
        if force_use_environment:
            sep = '\\\\\n'
            latex_eq = f"\\begin{{align*}} {sep.join(eqs)} \\end{{align*}}"
            display(Math(latex_eq))

    def pprint_equations(self, display_full_equations: bool = False):
        """
        Pretty prints the equations.

        Parameters
        ----------
        display_full_equations : 
            If True, displays the the equation with explicitly written photokinetic factor.
        """

        if self.symbols['equations'] is None:
            return

        eqs = self.symbols['equations'] if display_full_equations else self.symbols['equations_Fk']


        if len(self.absorbing_compartments.values()) > 0:
            display(self.flux_equations[self.Flux_types.index(self.flux_type)], self.explicit_Fk_equation)

        for eq in eqs:
            display(eq)

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
                            subs: Union[None, Iterable[tuple]] = None,
                            print_solution: bool = True):
        """
        Performs the steady state approximation for the given species.

        Parameters
        ----------
        compartments : 
            List/tuple of compartments given as strings for which steady state
            approximation will be perfomerd. Names of compartments must be as
            those used for constructing the model.
        subs : 
            List of tuples. In each tuple, first entry is the old expression and second
            entry is the new expression. Default is None.
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

        for comp, eq, f in zip(all_compartments, self.symbols['equations_Fk'], self.symbols['compartments']):
            if comp in compartments:  # use SS approximation
                eq2solve.append(Eq(eq.rhs, 0))
                variables.append(f)
            else:  # keep the diff equation
                eq2solve.append(eq) 
                variables.append(f.diff(self.symbols['time']))

        solution = solve(eq2solve, variables)

        if len(solution) == 0:
            raise ValueError('Steady state solution for the given input does not exist.')

        # gather all symbols that will be used for substitution
        if subs:
            self.symbols['substitutions'].clear()
            free_symbols = set()
            for old, new in subs:
                free_symbols = free_symbols.union(new.free_symbols)
            # sort the symbols by their name
            self.symbols['substitutions'] = sorted(free_symbols, key=lambda symbol: symbol.name)

        if print_solution and len(self.absorbing_compartments.values()) > 0:
            display(self.flux_equations[self.Flux_types.index(self.flux_type)], self.explicit_Fk_equation)

        # the order of solutions is the same as the input
        for var, expr in solution.items():

            ## need to change as the orderning in solution changed
            comp = all_compartments[variables.index(var)]

            # make substitutions
            if subs:
                for old, new in subs:
                    expr = expr.subs(old, new)

            # simplify and factor
            expr = factor(simplify(expr))

            # make substitutions again, but now on factored equations
            if subs:
                for old, new in subs:
                    expr = expr.subs(old, new)


            # substitute Fk for explicit photokinetic factor
            # if n_abs > 0:
            #     expr_full = expr.subs(self.symbols['Fk'], self.symbols['explicit_Fk'])
            #     eq_full = Eq(var, expr_full)  # create equation
            # else:
            #     eq_full = Eq(var, expr)  # create equation

            eq_Fk = Eq(var, expr)  # create equation

            if comp in compartments:
                self.last_SS_solution['SS_eqs'][comp] = eq_Fk
            else:
                self.last_SS_solution['diff_eqs'][comp] = eq_Fk

            if print_solution:
                # if print_full_equations:
                #     display(eq_full)
                # else:
                
                display(eq_Fk)

    def clear_model(self):
        """Clears the model."""

        self.symbols['compartments'].clear()
        self.symbols['equations'].clear()
        self.symbols['equations_Fk'].clear()
        self.symbols['rate_constants'].clear()
        self.symbols['time'] = None
        self.symbols['flux'] = None
        self.symbols['Fk'] = None
        self.symbols['l'] = None
        self.symbols['epsilons'].clear()
        self.symbols['substitutions'].clear()

        self.absorbing_compartments.clear()
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
        # self.symbols['time'], self.symbols['flux'],  self.symbols['Fk'] = symbols('t J F_k')
        self.symbols['time'] = symbols('t')
        self.symbols['l'] = symbols('l')
        self.symbols['flux'] = Function('J')(self.symbols['time'])
        self.symbols['Fk'] = Function('F_k')(self.symbols['time'])

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

        # abs_comp_idx = None
        # n_abs = len(list(filter(lambda el: el['type'] == 'absorption', self.elem_reactions)))

        # absorbing_compartments_idxs = list(filter(lambda el: el['type'] == 'absorption', self.elem_reactions))
        # absorbing_compartments_idxs  = list(map(lambda el: inv_idx[el['from_comp'][0]], absorbing_compartments_idxs))
        # n_abs = len(absorbing_compartments_idxs)
        # sum_abs_comps = None

        # for idx in absorbing_compartments_idxs:
        #     self.symbols['epsilons'].append(symbols(f'epsilon_{{{comps[idx]}}}'))

        n_abs = 0  # number of absorbing compartments
        eps_conc_products = []

        for el in self.elem_reactions:
            i_from = list(map(lambda com: inv_idx[com], el['from_comp']))  # list of indexes of starting materials
            i_to = list(map(lambda com: inv_idx[com], el['to_comp']))  # list of indexes of reaction products

            if el['type'] == 'absorption':
                n_abs += 1
                _from = i_from[0]
                _to = i_to[0]

                # create new epsilon symbol for new absorbing compartment
                eps = symbols(f'\\varepsilon_{{{comps[_from]}}}')
                self.symbols['epsilons'].append(eps)
                self.absorbing_compartments[comps[_from]] = _from
                
                expr = self.symbols['flux'] * self.symbols['Fk'] * self.symbols['compartments'][_from] * eps

                sympy_rhss[_from] -= expr
                sympy_rhss[_to] += expr

                eps_conc_products.append(self.symbols['compartments'][_from] * eps)

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

            eq_Fk = Eq(f.diff(self.symbols['time']), rhs)
            eq_full = Eq(f.diff(self.symbols['time']), rhs)

            # substitute Fk for full photokinetic factor
            if n_abs > 0:
                A_prime = sum(eps_conc_products)
                Fk = (1 - 10 ** (-self.symbols['l'] * A_prime)) / A_prime
                self.symbols['explicit_Fk'] = Fk
                self.explicit_Fk_equation = Eq(self.symbols['Fk'], Fk)  
                eq_full = eq_full.subs(self.symbols['Fk'], Fk)

            self.symbols['equations'].append(eq_full)
            self.symbols['equations_Fk'].append(eq_Fk)

        self.create_flux_equations()

    def get_parameter_map(self) -> dict[str, Symbol]:
        """
        Returns a dictionary of parameters for the model. The keys are the parameter names, and the values are dictionaries
        containing the parameter symbol, value, length, unit, latex name, and scaling function.
        """
        d = {}

        def add_entry(key: str, symbol: Symbol, value: float | Iterable[float] = None, 
                      length: int = 1, unit: str = None, latex_name: str = None, scale_data: Callable = None):
            d[key] = dict(symbol=symbol, value=value, length=length, unit=unit, latex_name=latex_name, scale_data=scale_data)

        # remove curly braces from the compartment names
        def clean_key(key: str) -> str:
            """Remove curly braces and backslashes from the key."""
            return key.replace('{', '').replace('}', '').replace('\\', '')
        
        for s in self.symbols['compartments']:  # initial concentrations
            add_entry(clean_key(f"{str(s)[:-3]}_0"), s, unit=f"\\mathrm{{{self.concentration_unit}}}", latex_name=f"[\\mathrm{{{str(s)[2:-3]}}}]_0",
                      scale_data=lambda x, coef: x / coef)

        for i, s in enumerate(self.symbols['rate_constants']):
            conc_unit = f'\\mathrm{{{self.concentration_unit}}}^{{{1 - self._rate_constant_orders[i]}}}\\ ' if self._rate_constant_orders[i] > 1 else ''
            add_entry(clean_key(str(s)), s, unit=f'\\mathrm{{{conc_unit}s^{{-1}}}}', latex_name=str(s),
                      scale_data=lambda x, coef: x * coef ** (self._rate_constant_orders[i] - 1))

        for s in self.symbols['epsilons']:
            add_entry(clean_key(f"{str(s)[4:]}"), s, unit=f"\\mathrm{{{self.concentration_unit}}}^{{-1}}\\ \\mathrm{{cm}}^{{-1}}", latex_name=str(s),
                      scale_data=lambda x, coef: x * coef)

        for s in self.symbols['substitutions']:
            add_entry(clean_key(str(s)), s, unit='', latex_name=str(s),
                      scale_data=lambda x, coef: x)

        add_entry('l', self.symbols['l'], unit='\\mathrm{cm}', latex_name='l',
                  scale_data=lambda x, coef: x)

        # self.symbols['other_symbols'] = [sw, J0, FWHM]
        add_entry('s_w', self.symbols['other_symbols'][0], unit="\\mathrm{s}", latex_name='s_w',
                  scale_data=lambda x, coef: x)
        add_entry('J_0', self.symbols['other_symbols'][1], unit=f"\\mathrm{{{self.concentration_unit}}}\\ \\mathrm{{s}}^{{-1}}", latex_name='J_0',
                  scale_data=lambda x, coef: x / coef)
        add_entry('FWHM', self.symbols['other_symbols'][2], unit="\\mathrm{s}", latex_name='FWHM',
                  scale_data=lambda x, coef: x)

        return d

    def get_par_dict(self) -> dict[str, float | None]:
        """
        Returns a dictionary of parameters for the model.
        """

        params = {key: None for key in self.get_parameter_map().keys()}
        params['l'] = 1
        params['J_0'] = 1e-5
        params['FWHM'] = 1
        params['s_w'] = 1

        return params

    def plot_simulation_results(self,  plot_separately: bool = True, t_unit: str = 's', 
                          yscale: str = 'linear',   figsize: tuple = (6, 3), 
                          cmap: str = 'plasma', lw: float = 2, 
                          legend_fontsize: int = 10, legend_labelspacing: float = 0,
                          legend_loc: str = 'center right', legend_bbox_to_anchor: tuple = (1.5, 0.5),
                          stack_plots_in_rows: bool = True, filepath: str = None, 
                          dpi: int = 300, transparent: bool = False,
                          auto_convert_time_units: bool = True,
                          sig_figures: int = 3, include_absorbance: bool = True):
        """
        Plots the results of the simulation. The function creates either separate plots for each compartment
        or combines all compartments into a single plot, depending on the plot_separately parameter.

        Parameters
        ----------
        t_unit : str
            Time unit to display on the x-axis. Default 's'.
        yscale : str
            Scale of the y axis. Default 'linear'. Can be 'log', etc.
        plot_separately: 
            If True (default True), the plot will have n graphs in which n is number of compartments to plot. In each graph,
            the one compartment will be plotted. If there are some iterables (length k) in the input parameters, each graph
            will have k curves.
            If True, rate constant, initial concentrations and epsilon and flux will be scaled by suitable factor
            (determined by the geometric mean of the initial concentrations). This can help to reduce numerical errors
            in the integrator. Optional. Default True.
        figsize : tuple
            Figure size of one graph. Default (6, 3): (width, height).
        cmap : str
            Color map used to color the individual curves if plot_separately = True. Default 'plasma'.
        lw : float
            Line width of the curves. Default 2.
        legend_fontsize : int
            Fontsize of the legend. Default 10.
        legend_labelspacing : float
            Vertical spacing of the labels in legend. Default 0. Can be negative.
        legend_loc : str
            Location of the legend. Default 'best'.
        legend_bbox_to_anchor : tuple
            Box to position the legend. Default None.
        stack_plots_in_rows : bool
            If True (default), stacks plots vertically. If False, arranges them horizontally.
        filepath : str
            If specified, saves the plot to this location. Default None.
        dpi : int
            DPI of the resulting plot that will be saved to file. Default 300.
        transparent : bool
            If True, the saved image will be transparent. Default False.
        auto_convert_time_units : bool
            If True (default), time will be converted to corresponding smaller or bigger units.
            Time unit will be denoted on x axis.
        sig_figures : int
            Number of significant figures to display in the legend. Default 3.
        include_absorbance : bool
            If True (default), the absorbance will be plotted for plot_separately = True only.
        """

        k = self.last_parameter_matrix.shape[0]

        comps = self.get_compartments()
        idxs2plot = [comps.index(comp) for comp in self.compartments_to_plot]

        comps_cast = self.compartments_to_plot
        traces_cast = [self.C_tensor[:, i, :] for i in idxs2plot]

        params_map = self.last_parameter_map

        par_names = []

        for i in range(k):
            entry = []

            for key, d in params_map.items():
                if d['length'] == 1:
                    continue
                entry.append(f"{d['latex_name']}={format_number_latex(d['value'][i], sig_figures)}\\ {d['unit']}")

            par_names.append(',\\ '.join(entry))

        times_scaled = self.times.copy()
        if auto_convert_time_units:
            s_factor, t_unit = get_time_unit(self.times[-1])
            times_scaled *= s_factor

        n_abs = len(self.absorbing_compartments)

        # Prepare data for plotting
        num_J = 1 if self.flux_type != 'Continuous' else 0
        num_A = 1 if n_abs > 0 and include_absorbance else 0

        def setup_axis(ax, yscale):
            ax.set_yscale(yscale)
            ax.tick_params(axis='both', which='major', direction='in')
            ax.tick_params(axis='both', which='minor', direction='in')
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')

        if plot_separately:
            n_rows = len(self.compartments_to_plot) + num_J + num_A
            figsize = (figsize[0], figsize[1] * n_rows) if stack_plots_in_rows else (figsize[0] * n_rows, figsize[1])
            fig, axes = plt.subplots(n_rows if stack_plots_in_rows else 1, 1 if stack_plots_in_rows else n_rows, figsize=figsize, sharex=True)

            cmap = cm.get_cmap(cmap)
            if num_J:
                ax = axes[0]
                for j in range(k):
                    ax.plot(times_scaled, self.J(self.times, j), label='', lw=lw, color=cmap(j / k))
                ax.set_title('$J(t)$')
                ax.set_ylabel('$J(t)$')
                setup_axis(ax, yscale)

            if num_A:
                ax = axes[num_J]
                for j in range(k):
                    ax.plot(times_scaled, self.A_tensor[j], label='', lw=lw, color=cmap(j / k))
                ax.set_title('Total absorbance')
                ax.set_ylabel('Absorbance')
                setup_axis(ax, yscale)

            plots = []

            for i, (comp, trace, ax) in enumerate(zip(comps_cast, traces_cast, axes[num_J + num_A:])):
                plots = []
                for j in range(k):
                    label = '' if par_names[j] == '' else f'${par_names[j]}$'
                    line, = ax.plot(times_scaled, trace[j], label=label, lw=lw, color=cmap(j / k))
                    plots.append(line)
                ax.set_ylabel(f'c / {self.concentration_unit}')
                # if i == (0 if stack_plots_in_rows else n_rows - 1) and k > 1:
                #     ax.legend(frameon=False, fontsize=legend_fontsize, labelspacing=legend_labelspacing, loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)
                ax.set_title(f'$\\mathrm{{{comp}}}$')
                setup_axis(ax, yscale)
            if k > 1:
                fig.legend(handles=plots, bbox_to_anchor=legend_bbox_to_anchor, 
                           fontsize=legend_fontsize, labelspacing=legend_labelspacing, loc=legend_loc)
            if stack_plots_in_rows:
                axes[-1].set_xlabel(f'Time / ${t_unit}$')
            else:
                for ax in axes:
                    ax.set_xlabel(f'Time / ${t_unit}$')

        else:
            figsize = (figsize[0], figsize[1] * k)
            fig, axes = plt.subplots(k if stack_plots_in_rows else 1, 1 if stack_plots_in_rows else k, figsize=figsize, sharex=True)

            for i in range(k):
                ax = axes[i] if k > 1 else axes

                for j, (comp, trace) in enumerate(zip(comps_cast, traces_cast)):
                    color = COLORS[j % len(COLORS)]
                    ax.plot(times_scaled, trace[i], label=f'$\\mathrm{{{comp}}}$', lw=lw, color=color)

                if i == k - 1:
                    ax.set_xlabel(f'Time / ${t_unit}$')
                ax.set_ylabel(f'c / {self.concentration_unit}')
                title = '' if par_names[i] == '' else f'${par_names[i]}$'
                ax.set_title(title)
                ax.set_yscale(yscale)
                ax.tick_params(axis='both', which='major', direction='in')
                ax.tick_params(axis='both', which='minor', direction='in')
                ax.xaxis.set_ticks_position('both')
                ax.yaxis.set_ticks_position('both')
                ax.legend(frameon=False, fontsize=legend_fontsize, labelspacing=legend_labelspacing, loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)
            
        plt.tight_layout()
        if filepath:
            plt.savefig(filepath, dpi=dpi, transparent=transparent)
        plt.show()

    def simulate_model(self, parameters: dict[str, Union[float, Iterable]],
                       constant_compartments: Union[None, List[str], Tuple[str]] = None,
                       t_max: Union[float, int] = 1e3, t_points: int = 1000, 
                       use_SS_approx: bool = True, ODE_solver: str = 'Radau', 
                       rescale: bool = True, default_max_step: float = np.inf):
        """
        Simulates the current model and plots the results if ``plot_results`` is True (default True). Parameters
        rate_constant and initial_concentrations can contain iterables. In this case, the model will be simulated
        for each of the value in the iterable arrays. Multiple iterables are allowed. In such case, lengths of
        iterable arrays in parameters rate_constants and intial_concentrations has to be equal. The result of
        the simulation is saved in a class attribute C_tensor.

        Parameters
        ----------
        parameters: dict[str, Union[float, Iterable]]
            Dictionary of parameters. The keys are the names of the parameters and the values are the values of the parameters.
            The values can be floats or iterables. If iterables, the model will be simulated for each of the value in the iterable arrays.
        constant_compartments: list of strings
            If specified (default None), the values of these compartments will be constant and the value from
            initial_concentration array will be used. Constant compartments will not be plotted. If the SS 
            approximation was performed for the constant compartment, it will have not effect and the SS 
            solution will be used instead.
        t_max: 
            Maximum time in seconds in the range of time points the model will be simulated for. Optional. Default 1e3 s.
        t_points: 
            Number of time points used to simulate the model for. Optional. Default 1000.
        use_SS_approx: 
            If True (default True), the equations from SS approximation will be used to simulate the model
            (only if it was performed for the model).
        ODE_solver: 
            Name of the ODE solver used to numerically simulate the model. Default 'Radau'. Available options are:
            'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'. For details, see
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
        sig_figures : 
            Number of significant figures the rate constants, initial concentrations and/or substitutions will be rounded to 
            if displayed in the plot.
        default_max_step:
            The default maximum step size for the ODE solver. Used when the pulse duration is not specified 
            and the integration is done over the entire time range.
        rescale:
            If True (default True), the parameters will be rescaled to have comparable values. This can help to reduce numerical errors
            in the integrator.
        """
        
        if len(self.symbols['equations']) == 0:
            raise ValueError("No model instantiated.")

        comps = self.get_compartments()
        n = len(comps)

        params_map = self.get_parameter_map()
        inv_params_map = {v['symbol']: k for k, v in params_map.items()}

        # symbols for lamdify functions
        # remove the symbols that are not needed for the lamdify functions
        # initial concentrations became the real concentrations and are storred at the beginning of the dictionary
        # remove l, J_0, FWHM, s_w
        init_conc_names = [key for key in list(params_map.keys())[:n]]
        init_conc_symbols = [params_map[key]['symbol'] for key in init_conc_names]

        symbols_lamdify_dict = params_map.copy()
        # remove first n entries to remove initial concentrations
        symbols_lamdify_dict = {key: symbols_lamdify_dict[key] for key in list(symbols_lamdify_dict.keys())[n:]}
        del symbols_lamdify_dict['l']
        del symbols_lamdify_dict['J_0']
        del symbols_lamdify_dict['FWHM']
        del symbols_lamdify_dict['s_w']

        # create an index map that will map from par_matrix to symbols_lamdify
        sim_key_list = list(params_map.keys())
        sym_map_idxs = list(map(sim_key_list.index, list(symbols_lamdify_dict.keys())))

        # symbols_lamdify_dict[str(self.symbols['flux'])] = self.symbols['flux']
        # symbols_lamdify_dict[str(self.symbols['Fk'])] = self.symbols['Fk']

        symbols_lamdify = init_conc_symbols + list(map(lambda entry: entry['symbol'], symbols_lamdify_dict.values())) + [self.symbols['flux'], self.symbols['Fk']]

        n_abs = len(self.symbols['epsilons'])
        use_SS_approx = use_SS_approx and len(self.last_SS_solution['SS_eqs'].values()) > 0

        params = self.get_par_dict()

        for key, value in parameters.items():
            if key in params.keys():
                params[key] = value

        not_specified = []
        for key, value in params.items():
            if value is None:
                not_specified.append(key)

        assert len(not_specified) == 0, f"Parameters {not_specified} must be specified."
    
        # find the maximal length of params which are iterable
        k = max(list(map(lambda v: len(v) if np.iterable(v) else 1, params.values())))
        par_matrix = np.zeros((k, len(params)))

        par_matrix_index_dict = {key: i for i, key in enumerate(params.keys())}
        # inv_par_matrix_index_dict = {i: key for i, key in enumerate(params.keys())}

        for i, (key, value) in enumerate(params.items()):
            if np.iterable(value):
                assert len(value) == k, f"Length of {key} must match the length of all iterables."
                par_matrix[:, i] = np.asarray(value)
                params_map[key]['length'] = len(value)
            else:
                par_matrix[:, i] = value
                params_map[key]['length'] = 1
            params_map[key]['value'] = value

        if any((par_matrix < 0).flatten()):
            raise ValueError("All parameters must be positive.")

        coef = 1
        par_matrix_scaled = par_matrix.copy()
        get_pars = lambda name: par_matrix_scaled[:, par_matrix_index_dict[name]]
        # get_number_of_params = lambda name: len(params[name]) if np.iterable(params[name]) else 1

        init_c = par_matrix[:, :n]   # initial concentrations

        # scale the parameters to have comparable values and to help with integration
        if rescale:
            # scaling coefficient, calculate it as geometric mean from non-zero initial concentrations
            coef = gmean(init_c[init_c > 0])
            for i, (key, d) in enumerate(params_map.items()):
                par_matrix_scaled[:, i] = d['scale_data'](par_matrix[:, i], coef)

            init_c /= coef

            # # print(coef)
            # init_c_sc /= coef
            # eps_sc *= coef
            # J0_sc /= coef
            # # second and higher orders needs to be appropriately scaled, by coef ^ (rate order - 1)
            # rates_sc *= coef ** (np.asarray(self._rate_constant_orders, dtype=dtype) - 1)

        self.last_parameter_matrix = par_matrix
        self.last_parameter_map = params_map

        dtype = np.float64

        J0s = get_pars('J_0')

        self.C_tensor = None
        if self.flux_type == 'Gaussian pulse':
            assert n_abs > 0, "Gaussian pulse is not allowed for the model without absorbing compartments."
            fwhms = get_pars('FWHM')
            times = np.linspace(-fwhms.max() * 3, t_max, int(t_points), dtype=dtype)
            pulse_duration = min(6 * fwhms.max(), t_max)  # for ivp solver
            max_step = min(fwhms.min() / 20, t_max / 10)
            self.J = lambda t, i: gaussian(t, fwhms[i], J0s[i])
        elif self.flux_type == 'Square pulse':
            assert n_abs > 0, "Square pulse is not allowed for the model without absorbing compartments."
            times = np.linspace(0, t_max, int(t_points), dtype=dtype)
            pulse_duration = None
            max_step = np.inf
            sw = get_pars('s_w')
            self.J = lambda t, i: square(t, sw[i], J0s[i])
        else:
            times = np.linspace(0, t_max, int(t_points), dtype=dtype)
            pulse_duration = None
            max_step = np.inf
            self.J = lambda t, i=None: J0s[i] * np.ones_like(t) if isinstance(t, np.ndarray) else J0s[i]


        self.times = times

        # differential equations to be simulated
        sym_eqs = list(self.last_SS_solution['diff_eqs'].values()) if use_SS_approx else self.symbols['equations_Fk']

        # those compartments that will be simulated numerically
        idxs2simulate = list(map(comps.index, self.last_SS_solution['diff_eqs'].keys())) if use_SS_approx else list(np.arange(n, dtype=int))

        idxs_constant = []  # empty list
        idxs_constant_cast = []
        if constant_compartments is not None:
            idxs_constant = map(comps.index, filter(lambda c: c in comps, constant_compartments))
            idxs_constant_cast = list(map(idxs2simulate.index, filter(lambda c: c in idxs2simulate, idxs_constant)))

        # get indexes of absorbing compartments
        # the same order as in epsilons
        idxs_abs = list(self.absorbing_compartments.values())

        # create a mapping of epsilon values from par_matrix
        eps_params_names = [inv_params_map[sym] for sym in self.symbols['epsilons']]
        eps_params_idxs = [par_matrix_index_dict[name] for name in eps_params_names]

        d_funcs = list(map(lambda eq: lambdify(symbols_lamdify, eq.rhs, 'numpy'), sym_eqs))

        l = get_pars('l')
        eps = par_matrix_scaled[:, eps_params_idxs]

        def dc_dt(t, c, i):  # index
            # only compartments in idxs2simulate are being simulated
            _c = np.zeros(n)  # need to cast the concentations to have the original size of the compartments
            _c[idxs2simulate] = c
 
            _eps = eps[i, :]

            # if we have absorbing compartments, we need to calculate the photokinetic factor
            # Approximate Fk, ignores the concentrations of steady state-compartments, c of them is set to 0 
            Fk = 0
            _J = 0
            if n_abs > 0:
                A_prime = np.sum(_eps * _c[idxs_abs])
                Fk = photokin_factor(A_prime, l[i])
                _J = self.J(t, i)

            solution = np.asarray([f(*_c, *par_matrix_scaled[i, sym_map_idxs], _J, Fk) for f in d_funcs])
            solution[idxs_constant_cast] = 0  # set the change of constant compartments to zero

            return solution

        C = np.zeros((k, n, times.shape[0]))

        if use_SS_approx:
            f_SS = list(map(lambda item: (comps.index(item[0]), lambdify(symbols_lamdify, item[1].rhs, 'numpy')), self.last_SS_solution['SS_eqs'].items()))

        for i in range(k):
            # numerically integrate

            C[i, idxs2simulate, :] = ode_integrate(dc_dt, init_c[i, idxs2simulate], times, method=ODE_solver, rtol=1e-6, atol=1e-9,
                    pulse_max_step=max_step, pulse_duration=pulse_duration, args=(i,), default_max_step=default_max_step)

            # calculate the SS concentrations from the solution of diff. eq.
            if use_SS_approx:
                # if we have absorbing compartments, we need to calculate the photokinetic factor
                # Approximate Fk, ignores the concentrations of steady state-compartments, c of them is set to 0 
                Fk = 0
                _J = 0
                if n_abs > 0:
                    A_prime = np.sum(eps[i, :, None] * C[i, idxs_abs, :], axis=0)
                    Fk = photokin_factor(A_prime, l[i])
                    _J = self.J(times, i)

                for j, f in f_SS:
                    C[i, j, :] = f(*list(C[i]), *par_matrix_scaled[i, sym_map_idxs], _J, Fk)

        # calculate the absorbance
        self.A_tensor = l[:, None] * np.sum(eps[:, :, None] * C[:, idxs_abs, :], axis=1)

        # scale the traces back to correct values
        C *= coef
        self.C_tensor = C

        # create indexes of only those comps. which will be plotted, so remove all constant comps.
        idxs2plot = set(np.arange(n, dtype=int)) - set(np.asarray(idxs2simulate)[idxs_constant_cast])
        self.compartments_to_plot = [comps[i] for i in idxs2plot]

    def plot_depenency(self, parameter_name: str, compartment_name: str, 
                      data_type: str = "last", yscale: str = "linear", plot_type: str = "scatter", figsize: tuple = (5, 4)):
        """
        Plots the dependency of the parameter on the compartment concentration at the beggining, last points or the integral of the compartment concentration over time.

        Parameters
        ----------
        parameter_name: str
            The name of the parameter to plot the dependency on.
        compartment_name: str
            The name of the compartment to plot the dependency on.
        data_type: str
            The type of data to plot. Can be "last", "integral" or "first".
        yscale: str
            The scale of the y-axis. Can be "linear" or "log".
        plot_type: str
            The type of plot to use. Can be "scatter" or "line".
        figsize: tuple
            The size of the figure. Default (6, 4).
        """

        assert self.C_tensor.shape[0] > 1, "No parameter was changed, no dependency can be plotted"

        comps = self.get_compartments()
        idx = comps.index(compartment_name)

        if data_type == "last":
            points = self.C_tensor[:, idx, -1]
        elif data_type == "first":
            points = self.C_tensor[:, idx, 0]
        elif data_type == "integral":
            points = np.trapz(self.C_tensor[:, idx, :], x=self.times, axis=1)
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

        par = self.last_parameter_map[parameter_name]
        if par['length'] > 1:
            x_vars = par['value']
        else:
            x_vars = par['value'] * np.ones(points.shape[0])

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if plot_type == "scatter":
            ax.scatter(x_vars, points, color='red', s=12)
        elif plot_type == "line":
            ax.plot(x_vars, points, color='red', lw=2)
        else:
            raise ValueError(f"Invalid plot_type: {plot_type}")

        ax.set_yscale(yscale)
        ax.tick_params(axis='both', which='major', direction='in')
        ax.tick_params(axis='both', which='minor', direction='in')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

        ax.set_xlabel(f'${par["latex_name"]}$')
        ax.set_ylabel(f'$c_{{\\mathrm{{{compartment_name}}}}}(t)$' + f' / {self.concentration_unit}')

        plt.tight_layout()

        plt.show()


    def print_text_model(self):
        """Print the model as text."""
        print(f'Scheme: {self.scheme}')

        for el in self.elem_reactions:
            rate = el['rate_constant_name']# if el['type'] == 'reaction' else None
            print(f"{el['type'].capitalize()}: {' + '.join(el['from_comp'])} \u2192 {' + '.join(el['to_comp'])}, "
                  f"rate: {rate}")  # {rate=} does not work in python 3.7 on colab



if __name__ == '__main__':


    text_model = """
    GS -hv-> ^1S --> GS  // k_s  # absorption and decay back to ground state
    ^1S --> P            // k_p   # formation of the photoproduct from the singlet state
    # ^1S -hv-> ^1S
    P -hv-> P
    """

    # instantiate the model
    model = PhotoKineticSymbolicModel.from_text(text_model)
    model.print_model()  # print the model

    model.flux_type = model.Flux_types[2]

    # model.pprint_equations()  # print the ODEs
    # model.pprint_equations(True)  # print the ODEs

    # model.steady_state_approx(['^1S'])

    # print(model.get_symbols_for_simulation_map())

    params = model.get_par_dict()
    print(params)

    sw = np.linspace(10, 100, 10)
    J0 = 1e-4 / sw

    J0 = np.linspace(1e-6, 1e-5, 10)

    params.update({
        'c_GS_0': 1e-5,
        'c_^1S_0': 0,
        'c_P_0': 0,
        'k_s': 1e9,
        'k_p': 1e8,
        'epsilon_GS': 1e5,
        'epsilon_P': 5e4,
        's_w': 1,
        'J_0': J0,
        'FWHM': 1e-10,
        'l': 1,
    })

    model.simulate_model(params, t_max=1e3,  ODE_solver="Radau", t_points=1e3, rescale=False)

    # print(model.last_parameter_map)

    model.plot_simulation_results(plot_separately=True)

    # model.plot_depenency('s_w', 'P', data_type="last", plot_type="scatter")

    # model.steady_state_approx(['^1S'])

    # print('\n')



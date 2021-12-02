
from sympy import Function, solve, Eq, factor, simplify, Symbol, symbols, lambdify
from IPython.display import display, Math
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from typing import List, Union, Tuple

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
def split_delimiters(text: str, delimiters: Union[list, tuple]) -> List[tuple]:
    """
    Splits the text with denoted delimiters and returns the list of tuples in which
    first entry is the splitted text and the 2nd is the delimiter.
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


class PhotoKineticSymbolicModel:
    """
    
    
    
    
    """

    delimiters = {
        'absorption': '-hv->',
        'reaction': '-->',
    }

    def __init__(self):
        # self.initial_conditions = {}
        self.elem_reactions = []  # list of dictionaries of elementary reactions
        self.scheme = ""
        self.last_SS_solution = dict(diff_eqs={}, SS_eqs={})  # contains dictionaries
        self._rate_constant_orders = []

        self.symbols = dict(compartments=[],
                            equations=[],
                            rate_constants=[],
                            time=None,
                            flux=None, 
                            l=None,
                            epsilon=None)

        self.last_simul_matrix = None # simulated traces

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

        Eg. absorption and formation of singlet state, triplet and then photoproducts which
        irreversibly reacts with the ground state
            GS -hv-> S_1 --> GS // k_S  # absorption and singlet state decay
            S_1 --> T_0 --> GS // k_{isc} ; k_T  # intersystem crossing and triplet decay
            T_0 --> P // k_P  # reaction from triplet state to form the products
            P + GS -->  // k_r  # products reacts irreversibly with the ground state

        :param scheme:
            input text-based model
        :return:
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
                    raise ValueError(f"Missing a species after absorption arrow ({cls.delimiters['absorption']}).")

                _model.add_elementary_reaction(rs, ps, type=r_type, rate_constant_name=r_name)

        _model.build_equations()

        return _model

    def add_elementary_reaction(self, from_comp=('A', 'A'), to_comp=('B', 'C'), type='reaction', rate_constant_name=None):
        """
        Adds the elementary reaction to the model. type can be either 'reaction' or 'absorption'.
        """
        from_comp = from_comp if isinstance(from_comp, (list, tuple)) else [from_comp]
        to_comp = to_comp if isinstance(to_comp, (list, tuple)) else [to_comp]

        el = dict(from_comp=from_comp, to_comp=to_comp, type=type, rate_constant_name=rate_constant_name)

        if el in self.elem_reactions:
            return

        self.elem_reactions.append(el)

    def print_model(self):
        if IN_COLAB:
            self.pprint_model_no_env()
        else:
            self.pprint_model_jupyter()

    def pprint_model_jupyter(self):
        """Pretty prints model. Will work only in Jupyter notebook or QtConsole environment.
        This uses the align latex environment."""

        latex_eq = ''

        for el in self.elem_reactions:
            reactants = ' + '.join([f'\\mathrm{{{comp}}}' for comp in el['from_comp']])
            products = ' + '.join([f'\\mathrm{{{comp}}}' for comp in el['to_comp']])

            r_name = el['rate_constant_name']

            latex_eq += f'{reactants} &\\xrightarrow{{{r_name}}} {products} \\\\'

        latex_eq = r'\begin{align}' + latex_eq + r'\end{align}'
        display(Math(latex_eq))

    def pprint_model_no_env(self):
        """Pretty prints model. Will work only in Jupyter notebook or QtConsole environment.
        No environment is used here."""

        for el in self.elem_reactions:
            reactants = ' + '.join([f'\\mathrm{{{comp}}}' for comp in el['from_comp']])
            products = ' + '.join([f'\\mathrm{{{comp}}}' for comp in el['to_comp']])

            r_name = el['rate_constant_name']

            latex_eq = f'{reactants} \\xrightarrow{{{r_name}}} {products}'

            display(Math(latex_eq))

    def pprint_equations(self, subs: List[tuple] = None):
        """Pretty prints the equations. Will work only in Jupyter notebook or QtConsole environment."""

        if self.symbols['equations'] is None:
            return

        for eq in self.symbols['equations']:
            _eq = eq
            if subs:
                for old, new in subs:
                    _eq = _eq.subs(old, new)

            display(_eq)

    def get_compartments(self):
        """
        Return the compartment names, the names are case sensitive.
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
                            subs: List[tuple] = None):
        """Performs the steady state approximation for the given species and displays the 
        result."""

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

            if subs:
                for old, new in subs:
                    eq = eq.subs(old, new)

            if comp in compartments:
                self.last_SS_solution['SS_eqs'][comp] = eq
            else:
                self.last_SS_solution['diff_eqs'][comp] = eq

            display(eq)

    def clear_model(self):
        self.symbols['compartments'].clear()
        self.symbols['equations'].clear()
        self.symbols['rate_constants'].clear()
        self.symbols['time'] = None
        self.symbols['flux'] = None
        self.symbols['l'] = None
        self.symbols['epsilon'] = None
        self._rate_constant_orders.clear()


    def build_equations(self):
        """Builds the equations. Converts the elementary reactions reperezentation to sympy differential equations."""

        self.clear_model()

        comps = self.get_compartments()

        # right hand side of diff. equations
        sym_rhss = len(comps) * [0]

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

        # r_names = list(map(lambda el: el['rate_constant_name'], filter(lambda el: el['type'] == 'reaction', self.elem_reactions)))
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

                sym_rhss[k] -= self.symbols['flux'] * fraction_abs

                k = i_to[0]
                sym_rhss[k] += self.symbols['flux'] * fraction_abs

                continue

            forward_prod = s_rates_dict[el['rate_constant_name']]

            for k in i_from:
                forward_prod *= self.symbols['compartments'][k]  # forward rate products, eg. k_AB * [A] * [B]

            for k in i_from:
                sym_rhss[k] -= forward_prod   # reactants

            for k in i_to:
                sym_rhss[k] += forward_prod   # products

        for f, rhs in zip(self.symbols['compartments'], sym_rhss):
            # construct differential equation
            _eq = Eq(f.diff(self.symbols['time']), rhs)
            self.symbols['equations'].append(_eq)

    def simulate_model(self, rate_constants: Union[List[float], Tuple[float], np.ndarray],
                       initial_concentrations: Union[List[float], Tuple[float], np.ndarray],
                       constant_compartments=Union[None, List[str], Tuple[str]],
                       t_max=1000, flux=1e-8, l=1, epsilon=1e5, t_points=1000, use_SS_approx: bool = True,
                       yscale='linear', scale=True, figsize=(8, 6)):

        """
            constant_compartments,  if specified, the concetration of those will be constant and value from 
            initial concentration will be used
            if use_SS_approx is True, 
        """
        
        if len(self.symbols['equations']) == 0:
            raise ValueError("No model instantiated.")

        comps = self.get_compartments()
        n = len(comps)
        use_SS_approx = use_SS_approx and len(self.last_SS_solution['SS_eqs'].values()) > 0
        init_c = np.asarray(initial_concentrations, dtype=np.float64)
        rates = np.asarray(rate_constants, dtype=np.float64)

        assert init_c.shape[0] == n
        assert rates.shape[0] == len(self.symbols['rate_constants'])

        self.last_simul_matrix = None

        # scale rate constants, flux, epsilon and initial_concentrations for less numerial errors in
        # the numerical integration
        if scale:
            coef = init_c[init_c > 0].min()  # scaling coefficient, lets choose the minimal concentration given
            init_c /= coef
            epsilon *= coef
            flux /= coef
            # second and higher orders needs to be appropriately scaled, by coef ^ (rate order - 1)
            rates *= coef ** (np.asarray(self._rate_constant_orders, dtype=np.float64) - 1)

        symbols = self.symbols['rate_constants'] + self.symbols['compartments'] + [self.symbols['flux'], self.symbols['l'], self.symbols['epsilon']]

        sym_eqs = list(self.last_SS_solution['diff_eqs'].values()) if use_SS_approx else self.symbols['equations']

        d_funcs = list(map(lambda eq: lambdify(symbols, eq.rhs, 'numpy'), sym_eqs))

        times = np.linspace(0, t_max, int(t_points))

        if use_SS_approx:
            idxs = list(map(comps.index, self.last_SS_solution['diff_eqs'].keys()))
            init_c = init_c[idxs]  # pick initial conditions only for those species that will be calculated by diff. eq.

        def dc_dt(c, t):
            if use_SS_approx:
                _c = np.zeros(n)  # need to cast the concentations to have the original size of the compartments
                _c[idxs] = c
            else:
                _c = c

            return [f(*rates, *_c, flux, l, epsilon) for f in d_funcs]

        # numerically integrate
        C = odeint(dc_dt, init_c, times)

        if use_SS_approx:
            tr_dict = dict(zip(comps, [None] * n))  # make compartments in the same order

            for comp, trace in zip(self.last_SS_solution['diff_eqs'].keys(), C.T):
                tr_dict[comp] = trace

            # calculate the SS concentrations from the solution of diff. eq.
            for comp, eq in self.last_SS_solution['SS_eqs'].items():
                f = lambdify(symbols, eq.rhs, 'numpy')
                tr_dict[comp] = f(*rates, *tr_dict.values(), flux, l, epsilon)
        else:
            tr_dict = dict(zip(comps, C.T))

        # combine traces
        for trace in tr_dict.values():
            self.last_simul_matrix = trace if self.last_simul_matrix is None else np.vstack((self.last_simul_matrix, trace))
        
        self.last_simul_matrix = self.last_simul_matrix.T  # transpose matrix

        # scale the traces back to correct values
        if scale:
            self.last_simul_matrix *= coef

        self.last_traces = tr_dict.values()


        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # plot the results
        for label, trace in zip(comps, self.last_simul_matrix.T):
            ax.plot(times, trace, label=f'$\\mathrm{{{label}}}$', lw=2.5)

        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration')
        ax.set_yscale(yscale)
        ax.legend(frameon=False)
        
        plt.show()

    def print_text_model(self):
        """Print the model as text."""
        print(f'Scheme: {self.scheme}')

        for el in self.elem_reactions:
            rate = el['rate_constant_name']# if el['type'] == 'reaction' else None
            print(f"{el['type'].capitalize()}: {' + '.join(el['from_comp'])} \u2192 {' + '.join(el['to_comp'])}, "
                  f"rate: {rate}")  # {rate=} does not work in python 3.7 on colab


if __name__ == '__main__':

    # model = """
    # ^1BR --> BR // k_S  # population of singlet state and decay to GS with rate k_S
    # ^1BR --> ^3BR --> BR -hv-> ^1BR  // k_{isc}; k_T
    # ^3BR + ^3O_2 --> ^1O_2 + BR  // k_{TT}
    # ^1O_2 --> ^3O_2  // k_d
    # BR + ^1O_2 --> // k_r
    
    # """

    model = """
    BR -hv-> ^1BR --> BR // k_S  # population of singlet state and decay to GS with rate k_S
    ^1BR -hv-> 
    """

    model = """
    A --> B  // k_1 
    2B --> //  k_2

    """
    # model = """
    # A -hv-> ^1A --> A           // k_d  # absorption and singlet state decay
    # ^1A -->  // k_r

    # """

    # model = """
    # ArO_2 --> Ar + ^1O_2             // k_1  # absorption and singlet state decay
    # ^1O_2 --> ^3O_2                  // k_d
    # # ^1O_2 + Ar --> Ar + ^3O_2         // k_{q,Ar}
    # # ^1O_2 + ArO_2 --> ArO_2 + ^3O_2     // k_{q,ArO_2}
    # ^1O_2 + S --> S + ^3O_2           // k_{q,S}
    # S + ^1O_2 -->                      // k_r
    # """


    model = PhotoKineticSymbolicModel.from_text(model)
    # print(model.print_text_model())

    model.simulate_model([1, 10], [0.1, 0], t_max=10, yscale='linear', scale=False)


    # model.steady_state_approx(['^1O_2'])
    # model.simulate_model([5e-3, 1/9.5e-6, 1e4, 1e9], [1e-3, 0, 0, 0, 1e-3], t_max=1e3, yscale='log', scale=True)







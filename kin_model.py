
from sympy import Function, solve, Eq, factor, simplify, Symbol, symbols
from IPython.display import display, Math
import re

import matplotlib.pyplot as plt

from typing import List, Union, Tuple

# import numpy as np
# import matplotlib.pyplot as plt
# import json
# from scipy.integrate import odeint

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
        self.last_SS_solution = []

        self.symbols = dict(compartments=[],
                            equations=[],
                            rate_constants=[],
                            time=None,
                            flux=None)

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
                sp_rates = list(map(lambda r: r.strip(), filter(None, rates[0].split(';'))))

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
                    print(r_name)

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

        self.last_SS_solution.clear()

        eq2solve = []
        variables = []

        for comp, eq, f in zip(self.get_compartments(), self.symbols['equations'], self.symbols['compartments']):
            if comp in compartments:
                eq2solve.append(Eq(eq.rhs, 0))
                variables.append(f)
            else:
                eq2solve.append(eq)
                variables.append(f.diff(self.symbols['time']))

        solution = solve(eq2solve, variables)

        for var, expression in solution.items():
            eq = Eq(var, expression)
            eq = factor(simplify(eq))

            if subs:
                for old, new in subs:
                    eq = eq.subs(old, new)

            self.last_SS_solution.append(eq)
            display(eq)

    def clear_model(self):
        self.symbols['compartments'].clear()
        self.symbols['equations'].clear()
        self.symbols['rate_constants'].clear()
        self.symbols['time'] = None
        self.symbols['flux'] = None

    def build_equations(self):
        """Builds the equations. Converts the elementary reactions reperezentation to sympy differential equations."""

        self.clear_model()

        comps = self.get_compartments()

        # right hand side of diff. equations
        sym_rhss = len(comps) * [0]

        # time and concentrations of absorbed photons J
        self.symbols['time'], self.symbols['flux'] = symbols('t J')

        for c in comps:
            # f = Function(f'[{{{c}}}]')(s_t)
            f = Function(f'c_{{{c}}}')(self.symbols['time'])
            self.symbols['compartments'].append(f)

        idx_dict = dict(enumerate(comps))
        inv_idx = dict(zip(idx_dict.values(), idx_dict.keys()))

        r_names = list(map(lambda el: el['rate_constant_name'], filter(lambda el: el['type'] == 'reaction', self.elem_reactions)))
        self.symbols['rate_constants'] = list(map(lambda n: Symbol(n), r_names))

        # symbolic rate constants dictionary
        s_rates_dict = dict(zip(r_names, self.symbols['rate_constants']))

        for el in self.elem_reactions:
            i_from = list(map(lambda com: inv_idx[com], el['from_comp']))  # list of indexes of starting materials
            i_to = list(map(lambda com: inv_idx[com], el['to_comp']))  # list of indexes of reaction products

            if el['type'] == 'absorption':
                for k in i_from:
                    sym_rhss[k] -= self.symbols['flux']

                for k in i_to:
                    sym_rhss[k] += self.symbols['flux']

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
    A -hv-> B
    2B + A --> 3C 
    2C -->   
    D --> 

    """

    str_model = """
    ArO_2 --> Ar + ^1O_2             // k_1  # absorption and singlet state decay
    ^1O_2 --> ^3O_2                  // k_d
    #  ^1O_2 + Ar --> Ar + ^3O_2         // k_{q,Ar}
    #  ^1O_2 + ArO_2 --> ArO_2 + ^3O_2     // k_{q,ArO_2}
    ^1O_2 + S --> S + ^3O_2           // k_{q,S}
    S + ^1O_2 -->                      // k_r
    """

    model = PhotoKineticSymbolicModel.from_text(str_model)
    print(model.print_text_model())







'''
Solve optimal firm location choice in a Stackelberg variant of the Hotelling's location choice model with N firms, where firm location choices can endogenously reduce the number of firms that enter the market if f is sufficiently large

AUTHORS: Kei Irizawa and Adam Alexander Oppenheimer
DATE: May 2020
'''

import numpy as np
from sympy import symbols, Eq, solve, diff, latex, integrate
import pandas as pd
from fractions import Fraction
import copy

import sys
sys.setrecursionlimit(10000)

def generate_symbols(N):
    '''
    Generate symbols

    To clarify variables:
        p: firm price
        i: firm location
        x: indifferent consumer location

    Inputs:
        N (int): number of firms

    Returns:
        all_symbols (list): list of dictionaries, where dictionaries contain: price symbols; location symbols; indifferent consumer symbols; and constants symbols
    '''
    price_symbols = {}
    location_symbols = {}
    indifferent_consumer_symbols = {}
    for i in range(1, N + 1):
        price_symbols['p_' + str(i)] = symbols('p_' + str(i))
        location_symbols['i_' + str(i)] = symbols('i_' + str(i))
        if i < N:
            indifferent_consumer_symbols['x_' + str(i)] = symbols('x_' + str(i))
    constant_symbols = {'c': symbols('c'), 't': symbols('t'), 'f': symbols('f'), 'V': symbols('V')}

    all_symbols = price_symbols, location_symbols, indifferent_consumer_symbols, constant_symbols

    return all_symbols

def analytic_profits_prices_cw(N_firms, all_symbols):
    '''
    Solve analytic profits (this involves first solving for indifferent consumers and optimal prices), prices, and consumer welfare

    To clarify variables:
        p: firm price
        i: firm location
        x: indifferent consumer location

    Inputs:
        N_firms (int): number of firms
        all_symbols (list): list of dictionaries, where dictionaries contain: price symbols; location symbols; indifferent consumer symbols; and constants symbols

    Returns:
        profit_functions (dictionary): dictionary linking firm to profit function
        price_functions (dictionary): dictionary linking firm to price
        cw_functions (dictionary): dictionary linking firm to cumulative consumer welfare for all consumers who purchase from that firm
    '''
    price_symbols, location_symbols, indifferent_consumer_symbols, constant_symbols = all_symbols

    # Solve the indifferent consumers
    indifferent_consumers = {}
    for i in range(1, N_firms):
        p_i = 'p_' + str(i)
        p_i_p1 = 'p_' + str(i + 1)
        i_i = 'i_' + str(i)
        i_i_p1 = 'i_' + str(i + 1)
        x_i = 'x_' + str(i)
        x_i_sln = solve(price_symbols[p_i] + constant_symbols['t'] * (indifferent_consumer_symbols[x_i] - location_symbols[i_i]) ** 2 - price_symbols[p_i_p1] - constant_symbols['t'] * (indifferent_consumer_symbols[x_i] - location_symbols[i_i_p1]) ** 2, indifferent_consumer_symbols[x_i])[0]
        indifferent_consumers[x_i] = x_i_sln

    # Generate the profit functions from the indifferent consumers
    profit_functions = {}
    pi_N = 'pi_' + str(N_firms)
    p_N = 'p_' + str(N_firms)
    x_N_m1 = 'x_' + str(N_firms - 1)
    profit_functions['pi_1'] = indifferent_consumers['x_1'] * (price_symbols['p_1'] - constant_symbols['c']) - constant_symbols['f']
    profit_functions[pi_N] = (1 - indifferent_consumers[x_N_m1]) * (price_symbols[p_N] - constant_symbols['c']) - constant_symbols['f']

    for i in range(2, N_firms):
        pi_i = 'pi_' + str(i)
        p_i = 'p_' + str(i)
        x_i_m1 = 'x_' + str(i - 1)
        x_i = 'x_' + str(i)
        profit_functions[pi_i] = (indifferent_consumers[x_i] - indifferent_consumers[x_i_m1]) * (price_symbols[p_i] - constant_symbols['c']) - constant_symbols['f']

    # Solve FOC for the profit functions
    FOC = {}
    for i in range(1, N_firms + 1):
        p_i = 'p_' + str(i)
        FOC[p_i] = Eq(diff(profit_functions['pi_' + str(i)], price_symbols[p_i]), 0)

    prices = solve(list(FOC.values()), list(price_symbols.values()))
    price_functions = {}

    # Now, substitute in for optimal prices in the firms' profit functions and indifferent consumers in the firms' prices
    for i in range(1, N_firms + 1):
        pi_i = 'pi_' + str(i)
        p_i = 'p_' + str(i)
        price_functions[p_i] = prices[price_symbols[p_i]].copy()
        for j in range(1, N_firms + 1):
            p_j = 'p_' + str(j)
            profit_functions[pi_i] = profit_functions[pi_i].subs(price_symbols[p_j], prices[price_symbols[p_j]])

    # Update indifferent consumers to include optimal prices
    for i in range(1, N_firms):
        x_i = 'x_' + str(i)
        for j in range(1, N_firms + 1):
            p_j = 'p_' + str(j)
            indifferent_consumers[x_i] = indifferent_consumers[x_i].subs(price_symbols[p_j], price_functions[p_j])

    # Finally, solve consumer welfare
    cw_functions = {}
    consumer_loc = symbols('x_i')
    firm_N = 'firm_' + str(N_firms)
    i_N = 'i_' + str(N_firms)
    cw_functions['firm_1'] = integrate(constant_symbols['V'] - price_functions['p_1'] - constant_symbols['t'] * (consumer_loc - location_symbols['i_1']) ** 2, (consumer_loc, 0, indifferent_consumers['x_1']))
    cw_functions[firm_N] = integrate(constant_symbols['V'] - price_functions[p_N] - constant_symbols['t'] * (consumer_loc - location_symbols[i_N]) ** 2, (consumer_loc, indifferent_consumers[x_N_m1], 1))

    for i in range(2, N_firms):
        firm_i = 'firm_' + str(i)
        p_i = 'p_' + str(i)
        i_i = 'i_' + str(i)
        x_i_m1 = 'x_' + str(i - 1)
        x_i = 'x_' + str(i)
        cw_functions[firm_i] = integrate(constant_symbols['V'] - price_functions[p_i] - constant_symbols['t'] * (consumer_loc - location_symbols[i_i]) ** 2, (consumer_loc, indifferent_consumers[x_i_m1], indifferent_consumers[x_i]))

    return profit_functions, price_functions, cw_functions

def calc_profits_prices_cw(i_vals, c_val, t_val, f_val, V_val, all_symbols, profit_functions, price_functions, cw_functions, N_firms):
    '''
    Solve profits and prices for a choice of locations

    Inputs:
        i_vals (list): sorted list (sorted by firm number) linking firms to firm location decisions taken as given
        c_val (float): variable cost
        t_val (float): cost of traveling
        f_val (float): fixed cost
        all_symbols (list of symbols): relevant Sympy symbols
        profit_functions (dictionary): dictionary linking firm to profit function
        price_functions (dictionary): dictionary linking firm to price
        cw_functions (dictionary): dictionary linking firm to cumulative consumer welfare for all consumers who purchase from that firm
        N_firms (int): number of firms

    Returns:
        pi_list (tuple of Fractions): tuple linking firm location to profit for each location
        price_list (tuple of Fractions): tuple linking firm location to prices for each location
        cw_list (tuple of Fractions): tuple linking firm location to cumulative consumer welfare for all consumers who purchase from the firm at each location
    '''
    price_symbols, location_symbols, constant_symbols = all_symbols[0], all_symbols[1], all_symbols[3]
    pi_list = []
    price_list = []
    cw_list = []

    for i in range(1, N_firms + 1):
        pi_i = 'pi_' + str(i) # For profits
        p_i = 'p_' + str(i) # For prices
        firm_i = 'firm_' + str(i) # For consumer welfare (consumers who purchase from firm i)

        pi_list.append(profit_functions[pi_i].subs(constant_symbols['c'], c_val))
        pi_list[i - 1] = pi_list[i - 1].subs(constant_symbols['t'], t_val)
        pi_list[i - 1] = pi_list[i - 1].subs(constant_symbols['f'], f_val)

        price_list.append(price_functions[p_i].subs(constant_symbols['c'], c_val))
        price_list[i - 1] = price_list[i - 1].subs(constant_symbols['t'], t_val)
        price_list[i - 1] = price_list[i - 1].subs(constant_symbols['f'], f_val)

        cw_list.append(cw_functions[firm_i].subs(constant_symbols['c'], c_val))
        cw_list[i - 1] = cw_list[i - 1].subs(constant_symbols['t'], t_val)
        cw_list[i - 1] = cw_list[i - 1].subs(constant_symbols['f'], f_val)
        cw_list[i - 1] = cw_list[i - 1].subs(constant_symbols['V'], V_val)

        for j in range(1, N_firms + 1):
            i_j = 'i_' + str(j)
            pi_list[i - 1] = pi_list[i - 1].subs(location_symbols[i_j], i_vals[j - 1])
            price_list[i - 1] = price_list[i - 1].subs(location_symbols[i_j], i_vals[j - 1])
            cw_list[i - 1] = cw_list[i - 1].subs(location_symbols[i_j], i_vals[j - 1])

    # Convert from Rational to Fraction
    for i, pi in enumerate(pi_list):
        pi_str = str(pi)
        pi_Frac = pi_str.split('/')
        if len(pi_Frac) == 2:
            pi_num, pi_denom = pi_Frac
            pi_list[i] = Fraction(int(pi_num), int(pi_denom))
    for i, price in enumerate(price_list):
        price_str = str(price)
        price_Frac = price_str.split('/')
        if len(price_Frac) == 2:
            price_num, price_denom = price_Frac
            price_list[i] = Fraction(int(price_num), int(price_denom))
    for i, firm in enumerate(cw_list):
        cw_str = str(firm)
        cw_Frac = cw_str.split('/')
        if len(cw_Frac) == 2:
            cw_num, cw_denom = cw_Frac
            cw_list[i] = Fraction(int(cw_num), int(cw_denom))


    return tuple(pi_list), tuple(price_list), tuple(cw_list)

def gen_pi_dict(i_vals, params, pi_dict, price_dict, cw_dict):
    '''
    Generate dictionary linking sorted tuples of all firm locations (sorted by location) to profits for each location

    Inputs:
        i_vals (list): sorted list (sorted by firm number, which is equivalent to sorting by firm location in this case) linking firms to firm location decisions taken as given
        params (list): various parameters
        pi_dict (dictionary): dictionary linking tuples of firm locations to profits for each location
        price_dict (dictionary): dictionary linking tuples of firm locations to prices for each location
        cw_dict (dictionary): dictionary linking tuples of firm locations to prices for each location

    Returns:
        Nothing
    '''
    c_val, t_val, f_val, V_val, all_symbols, profit_functions, prices, cw, N_firms, N_locations = params

    pi_dict[tuple(i_vals)], price_dict[tuple(i_vals)], cw_dict[tuple(i_vals)] = calc_profits_prices_cw(i_vals, c_val, t_val, f_val, V_val, all_symbols, profit_functions, prices, cw, N_firms)

    for i in range(N_firms):
        # Go backward through firms
        if i_vals[N_firms - i - 1] < 1 - Fraction(i, N_locations - 1):
            recursive_i_vals = i_vals.copy()
            recursive_i_vals[N_firms - i - 1] += Fraction(1, N_locations - 1)
            # Reset all future firms to closest locations
            for j in range(N_firms - i, N_firms):
                recursive_i_vals[j] = recursive_i_vals[j - 1] + Fraction(1, N_locations - 1)
            gen_pi_dict(recursive_i_vals, params, pi_dict, price_dict, cw_dict)
            break

def find_i_n_specific_path(i_vals, i_dicts_list, pi_dict, params, method, path_val):
    '''
    Interior method to find_i_n, only solve for a particular path

    Inputs:
        i_vals (list): sorted list (sorted by firm number, NOT firm location) linking firms to firm location decisions taken as given
        i_dicts_list (list of dictionaries): sorted list of dictionarires that link past firms' location decisions to future firms' optimal location decisions
        pi_dict (dictionary): dictionary linking tuples of firm locations to profits for each location
        params (list): various parameters
        method (string): 'expected value' or 'absolute' (absolute highest outcome for firm a)
        path_val (Fraction): specific path to take

    Returns:
        res (list): first entry contains either a) max firm profit (if method='absolute') or expected firm profit (if method='expected value'). Each following entry contains optimal firm n location, where multiple entries give indifferent solutions.
    '''
    c_val, t_val, f_val, V_val, all_symbols, profit_functions, prices, cw, N_firms, N_locations = params
    max_pi_n = False
    res = []
    # Look only at particular path
    i_n_val = path_val

    # First, solve future firms' decisions
    modified_i_vals = i_vals.copy()
    modified_i_vals.append(i_n_val)
    # Account for potential indifference for future firms between multiple decisions
    recursive_pi_list_weighted = []
    recursive_pi_list_unweighted = []
    cumulative_weight = 1
    for j, next_mover in enumerate(i_dicts_list):
        next_mover_decisions = next_mover[tuple(sorted(modified_i_vals))]
        if len(next_mover_decisions) == 1:
            modified_i_vals.append(next_mover_decisions[0])
        elif len(next_mover_decisions) == 0: # Fixed cost prevents future firm entry
            break # No more firms enter
        else:
            recursion_weight = Fraction(len(next_mover_decisions) - 1, len(next_mover_decisions))
            # Do some recursion to access sub-paths
            recursive_i_dicts_list = copy.deepcopy(i_dicts_list) # Only include firms starting where recursion applies
            recursive_i_dicts_list[j][tuple(sorted(modified_i_vals))] = recursive_i_dicts_list[j][tuple(sorted(modified_i_vals))][1:] # Drop 0, because using that result for this recursive loop
            recursive_res = find_i_n(i_vals, recursive_i_dicts_list, pi_dict, params, method)
            # If these recursion results have positive profits:
            if recursive_res:
                recursive_pi_list_weighted.append(cumulative_weight * recursion_weight * recursive_res[0])
                recursive_pi_list_unweighted.append(recursive_res[0])

            cumulative_weight *= (1 - recursion_weight)

            # Now continue with the normal loop
            modified_i_vals.append(next_mover_decisions[0])

    # Solve firm profits
    sorted_locations = sorted(modified_i_vals)
    firm_profits_unsorted = pi_dict[tuple(sorted_locations)]
    # Re-sort firm profits
    firm_profits_sorted = [firm_profits_unsorted[sorted_locations.index(i)] for i in modified_i_vals]

    pi_n = firm_profits_sorted[len(i_vals)]

    if method == 'absolute' and recursive_pi_list_unweighted:
        # If no recursive values, just leave pi_n
        pi_n = max(pi_n, max(recursive_pi_list_unweighted))
    elif method == 'expected value':
        pi_n = cumulative_weight * pi_n + sum(recursive_pi_list_weighted)

    if not max_pi_n:
        if pi_n >= 0:
            max_pi_n = pi_n
            res = [pi_n, i_n_val]
    elif pi_n > max_pi_n:
        max_pi_n = pi_n
        res = [pi_n, i_n_val]
    elif pi_n == max_pi_n:
        res.append(i_n_val)

    return res

def find_i_n(i_vals, i_dicts_list, pi_dict, params, method):
    '''
    Taking previous mover firms as given and calculating future mover firms decisions endogenously, find the value of i_n_val that maximizes firm n's profits. Solve recursively.

    Inputs:
        i_vals (list): sorted list (sorted by firm number, NOT firm location) linking firms to firm location decisions taken as given
        i_dicts_list (list of dictionaries): sorted list of dictionarires that link past firms' location decisions to future firms' optimal location decisions
        pi_dict (dictionary): dictionary linking tuples of firm locations to profits for each location
        params (list): various parameters
        method (string): 'expected value' or 'absolute' (absolute highest outcome for firm a)

    Returns:
        res (list): first entry contains either a) max firm profit (if method='absolute') or expected firm profit (if method='expected value'). Each following entry contains optimal firm n location, where multiple entries give indifferent solutions.
    '''
    c_val, t_val, f_val, V_val, all_symbols, profit_functions, prices, cw, N_firms, N_locations = params
    max_pi_n = False
    res = []
    for i in range(N_locations):
        i_n_val = Fraction(i, N_locations - 1) # Use N - 1 to ensure that bounds are [0, 1]
        if i_n_val not in i_vals: # Make sure firm n chooses an open location
            # First, solve future firms' decisions
            modified_i_vals = i_vals.copy()
            modified_i_vals.append(i_n_val)
            # Account for potential indifference for future firms between multiple decisions
            recursive_pi_list_weighted = []
            recursive_pi_list_unweighted = []
            cumulative_weight = 1
            for j, next_mover in enumerate(i_dicts_list):
                next_mover_decisions = next_mover[tuple(sorted(modified_i_vals))]
                if len(next_mover_decisions) == 1:
                    modified_i_vals.append(next_mover_decisions[0])
                elif len(next_mover_decisions) == 0: # Fixed cost prevents future firm entry
                    break # No more firms enter
                else:
                    recursion_weight = Fraction(len(next_mover_decisions) - 1, len(next_mover_decisions))
                    # Do some recursion to access sub-paths
                    recursive_i_dicts_list = copy.deepcopy(i_dicts_list) # Only include firms starting where recursion applies
                    recursive_i_dicts_list[j][tuple(sorted(modified_i_vals))] = recursive_i_dicts_list[j][tuple(sorted(modified_i_vals))][1:] # Drop 0, because using that result for this recursive loop
                    recursive_res = find_i_n_specific_path(i_vals, recursive_i_dicts_list, pi_dict, params, method, i_n_val)
                    # If these recursion results have positive profits:
                    if recursive_res:
                        recursive_pi_list_weighted.append(cumulative_weight * recursion_weight * recursive_res[0])
                        recursive_pi_list_unweighted.append(recursive_res[0])

                    cumulative_weight *= (1 - recursion_weight)

                    # Now continue with the normal loop
                    modified_i_vals.append(next_mover_decisions[0])

            # Solve firm profits
            sorted_locations = sorted(modified_i_vals)
            firm_profits_unsorted = pi_dict[tuple(sorted_locations)]
            # Re-sort firm profits
            firm_profits_sorted = [firm_profits_unsorted[sorted_locations.index(i)] for i in modified_i_vals]

            pi_n = firm_profits_sorted[len(i_vals)]

            if method == 'absolute' and recursive_pi_list_unweighted:
                # If no recursive values, just leave pi_n
                pi_n = max(pi_n, max(recursive_pi_list_unweighted))
            elif method == 'expected value':
                pi_n = cumulative_weight * pi_n + sum(recursive_pi_list_weighted)

            if not max_pi_n:
                if pi_n >= 0:
                    max_pi_n = pi_n
                    res = [pi_n, i_n_val]
            elif pi_n > max_pi_n:
                max_pi_n = pi_n
                res = [pi_n, i_n_val]
            elif pi_n == max_pi_n:
                res.append(i_n_val)

    return res

def gen_optimal_dicts(i_vals, i_dicts_list, pi_dict, params, method):
    '''
    Generate list of dictionaries giving optimal firm 

    Inputs:
        i_vals (list): sorted list (sorted by firm number, NOT firm location) linking firms to firm location decisions taken as given
        i_dicts_list (list of dictionaries): sorted list of dictionarires that link past firms' location decisions to future firms' optimal location decisions
        pi_dict (dictionary): dictionary linking tuples of firm locations to profits for each location
        params (list): various parameters
        method (string): 'expected value' or 'absolute' (absolute highest outcome for firm a)

    Returns:
        Nothing
    '''
    c_val, t_val, f_val, V_val, all_symbols, profit_functions, prices, cw, N_firms, N_locations = params

    opt_location = find_i_n(i_vals, i_dicts_list[1:], pi_dict, params, method) # First entry just gives profits, use i_dicts_list[1:] because not including this firm's decisions
    if opt_location:
        i_dicts_list[0][tuple(i_vals)] = opt_location[1:]
    else: # If no optimal location choice
        i_dicts_list[0][tuple(i_vals)] = opt_location

    no_break = True # Stays True if this is the last case for this firm
    for i in range(len(i_vals)):
        # Go backward through firms whose locations are taken as given
        if i_vals[len(i_vals) - i - 1] < 1 - Fraction(i, N_locations - 1):
            recursive_i_vals = i_vals.copy()
            recursive_i_vals[len(i_vals) - i - 1] += Fraction(1, N_locations - 1)
            # Reset all future firms to closest locations
            for j in range(len(i_vals) - i, len(i_vals)):
                recursive_i_vals[j] = recursive_i_vals[j - 1] + Fraction(1, N_locations - 1)
            gen_optimal_dicts(recursive_i_vals, i_dicts_list, pi_dict, params, method)
            no_break = False
            break

    if no_break and len(i_vals) > 0: # True if this is the last case for this firm, but this is not the last firm
        print('Solved for firm ', len(i_vals) + 1)
        i_dicts_list.insert(0, {})
        i_vals_init = []
        i_val = 0
        for i in range(len(i_vals) - 1): # Backward induction
            i_vals_init.append(i_val)
            i_val += Fraction(1, N_locations - 1)
        print('New i_vals:', i_vals_init)
        gen_optimal_dicts(i_vals_init, i_dicts_list, pi_dict, params, method)

def get_full_res(i_vals, i_list, i_dicts_list, pi_dict, price_dict, cw_dict, display_option='Fraction'):
    '''
    Print out results cleanly

    Inputs:
        i_vals (list): sorted list (sorted by firm number, NOT firm location) linking firms to firm location decisions taken as given
        i_list (list): list of locations that current firm is indifferent between choosing
        i_dicts_list (list of dictionaries): sorted list of dictionaries that link past firms' location decisions to future firms' optimal location decisions
        prev_i (list): list of locations that previous firms have chosen
        pi_dict (dictionary): dictionary linking tuples of firm locations to profits for each location
        price_dict (dictionary): dictionary linking tuples of firm locations to prices for each location
        cw_dict (dictionary): dictionary linking tuples of firm locations to prices for each location
        display_option (string): option whether to display profits as Fractions or floats

    Returns:
        full_res (list): list of full results, sorted by (locations, profits)
    '''

    full_res = []

    final_loop = False

    if len(i_dicts_list) > 0:
        next_firm_dict = i_dicts_list[0]
        next_firm_profitable = False
        for i in i_list:
            new_i_vals = i_vals.copy()
            new_i_vals.append(i)
            next_firm_locations = next_firm_dict[tuple(sorted(new_i_vals))]
            if next_firm_locations:
                i_res = get_full_res(new_i_vals, next_firm_dict[tuple(sorted(new_i_vals))], i_dicts_list[1:], pi_dict, price_dict, cw_dict, display_option)
                full_res.extend(i_res)
                next_firm_profitable = True
        if not next_firm_profitable:
            final_loop = True
    else:
        final_loop = True

    if final_loop:
        for i in i_list:
            full_i_vals = i_vals.copy()
            full_i_vals.append(i)

            sorted_locations = sorted(full_i_vals)
            firm_profits_unsorted = pi_dict[tuple(sorted_locations)]
            firm_prices_unsorted = price_dict[tuple(sorted_locations)]
            cw_unsorted = cw_dict[tuple(sorted_locations)]
            # Re-sort firm profits
            if display_option == 'Fraction':
                firm_profits_sorted = [firm_profits_unsorted[sorted_locations.index(i)] for i in full_i_vals]
                firm_prices_sorted = [firm_prices_unsorted[sorted_locations.index(i)] for i in full_i_vals]
                cw_sorted = [cw_unsorted[sorted_locations.index(i)] for i in full_i_vals]
            elif display_option == 'float':
                firm_profits_sorted = [float(firm_profits_unsorted[sorted_locations.index(i)]) for i in full_i_vals]
                firm_prices_sorted = [float(firm_prices_unsorted[sorted_locations.index(i)]) for i in full_i_vals]
                cw_sorted = [float(cw_unsorted[sorted_locations.index(i)]) for i in full_i_vals]

                #full_i_vals = [float(full_i_val) for full_i_val in full_i_vals]

            full_res.append((full_i_vals, firm_profits_sorted, firm_prices_sorted, cw_sorted))

    return full_res


def generate_plot(file_name, res):
    '''
    Generate tikz plots of firm locations and profits

    Inputs:
        file_name (string): file name
        res (tuple): sorted by firm, where first entry gives list of locations and  second entry gives list of profits
    '''

    colors = [r'red', r'blue', r'purple', r'green', r'orange', r'magenta']

    with open(file_name, 'w') as outfile:
        outfile.write(
            r'''\documentclass[margin=10pt]{standalone} 
            \usepackage{color,xcolor} 
            \usepackage{tikz-qtree, tikz} 
            \usepackage[utf8]{inputenc} 
            \usetikzlibrary{decorations.pathreplacing,arrows,shapes,positioning,shadows,calc}
            \usetikzlibrary{decorations, decorations.text,backgrounds}
            \tikzset{every picture/.style={font issue=\footnotesize}, font issue/.style={execute at begin picture={#1\selectfont}}}
            '''
        )

        outfile.write(
                r'''
                    \begin{document}
                        \centering
                            \begin{tikzpicture}

                            % Nodes 0 and 1
                            \draw (0,0) -- (15,0);
                            \draw (0 cm,3pt) -- (0 cm,-3pt);
                            \draw (15 cm,3pt) -- (15 cm,-3pt);
                '''
        )

        is_zero = False
        is_one = False

        for i, loc in enumerate(res[0]):
            output_str = r'% Firm' + str(i) + r'Choice' + '\n'
            outfile.write(output_str)
            
            output_str = r'\draw [' + colors[i] + r'] (' + str(loc) + r' * 15 cm,3pt) -- (' + str(loc) + r' * 15 cm,-3pt);' + '\n'
            outfile.write(output_str)

            output_str = r'\draw (' + str(loc) + r' * 15 cm,0) node[above=3pt, ' + colors[i] + r'] {$ F_' + str(i + 1) + r'$} node[above=3pt] {$ $};' + '\n'
            outfile.write(output_str)

            if (loc != 0) and (loc != 1):
                output_str = r'\draw (' + str(loc) + r' * 15 cm,0) node[below=3pt, ' + colors[i] + r'] {\tiny $ i_' + str(i + 1) + r'=\frac{' + str(loc.numerator) + r'}{' + str(loc.denominator) + r'}$} node[above=3pt] {$ $};' + '\n'
            else:
                if loc == 0:
                    is_zero = True
                elif loc == 1:
                    is_one = True
                output_str = r'\draw (' + str(loc) + r' * 15 cm,0) node[below=3pt, ' + colors[i] + r'] {\tiny $ i_' + str(i + 1) + r'=' + str(loc.numerator) + r'$} node[above=3pt] {$ $};' + '\n'

            outfile.write(output_str)

            output_str = r'\draw (' + str(loc) + r' * 15 cm,0) node[below=18pt, ' + colors[i] + r'] {\tiny $ \pi_' + str(i + 1) + '=' + '{:1.4f}'.format(res[1][i]) + r'$} node[above=3pt] {$ $};' + '\n'
            outfile.write(output_str)

            output_str = r'\draw (' + str(loc) + r' * 15 cm,0) node[below=33pt, ' + colors[i] + r'] {\tiny $ p_' + str(i + 1) + '=' + '{:1.4f}'.format(res[2][i]) + r'$} node[above=3pt] {$ $};' + '\n'
            outfile.write(output_str)

            output_str = r'\draw (' + str(loc) + r' * 15 cm,0) node[below=48pt, ' + colors[i] + r'] {\tiny $ CW_' + str(i + 1) + '=' + '{:1.4f}'.format(res[3][i]) + r'$} node[above=3pt] {$ $};' + '\n'
            outfile.write(output_str)

        if not is_zero:
            outfile.write(
                r'''
                \draw (0,0) node[below=3pt] {$ 0 $} node[above=3pt] {$ $};
                '''
            )
        if not is_one:
            outfile.write(
                r'''
                \draw (15,0) node[below=3pt] {$ 1 $} node[above=3pt] {$ $};
                '''
            )

        outfile.write(
            r'''
            \end{tikzpicture}
            \end{document}
            '''
        )

        outfile.close()

def run_model(inputs, method, plot=False):
    '''
    Run full model

    Inputs:
        inputs (list): list of all inputs
        method (string): 'exogenous': fixed number of firms, sets f_val=0; 'endogenous': endogenously determine number of firms
        plot (boolean): if True, generate plots

    Returns:
        full_res (list): list of full results, sorted by (locations, profits) (profits formatted as floats)
        full_res_2 (list): list of full results, sorted by (locations, profits) (profits formatted as Fractions)
        i_dicts_list (list of dictionaries): sorted list of dictionaries that link past firms' location decisions to future firms' optimal location decisions
        pi_dict (dictionary): dictionary linking tuples of firm locations to profits for each location
        price_dict (dictionary): dictionary linking tuples of firm locations to prices for each location
    '''

    N_firms, N_locations, c_val, t_val, f_val, V_val = inputs

    pi_dict = {}
    price_dict = {}
    cw_dict = {}

    if method == 'exogenous':
        print('Exogenous method')
        f_val = 0

        all_symbols = generate_symbols(N_firms)

        profit_functions, price_functions, cw_functions = analytic_profits_prices_cw(N_firms, all_symbols)

        params = c_val, t_val, f_val, V_val, all_symbols, profit_functions, price_functions, cw_functions, N_firms, N_locations

        i_vals_init = []
        i_val = 0
        for i in range(N_firms):
            i_vals_init.append(i_val)
            i_val += Fraction(1, N_locations - 1)

        gen_pi_dict(i_vals_init, params, pi_dict, price_dict, cw_dict)

    elif method == 'endogenous':
        print('Endogenous method')
        for N_firms_loop in range(2, N_firms + 1):
            all_symbols = generate_symbols(N_firms_loop)

            profit_functions, price_functions, cw_functions = analytic_profits_prices_cw(N_firms_loop, all_symbols)

            params = c_val, t_val, f_val, V_val, all_symbols, profit_functions, price_functions, cw_functions, N_firms_loop, N_locations

            i_vals_init = []
            i_val = 0
            for i in range(N_firms_loop):
                i_vals_init.append(i_val)
                i_val += Fraction(1, N_locations - 1)

            gen_pi_dict(i_vals_init, params, pi_dict, price_dict, cw_dict)

    i_vals_init_2 = []
    i_val = 0
    for i in range(N_firms - 1):
        i_vals_init_2.append(i_val)
        i_val += Fraction(1, N_locations - 1)

    i_dicts_list = [{}]

    gen_optimal_dicts(i_vals_init_2, i_dicts_list, pi_dict, params, method='expected value')

    res_list = find_i_n([], i_dicts_list[1:], pi_dict, params, method='expected value')

    full_res = get_full_res([], res_list[1:], i_dicts_list[1:], pi_dict, price_dict, cw_dict, display_option='float')

    full_res_2 = get_full_res([], res_list[1:], i_dicts_list[1:], pi_dict, price_dict, cw_dict, display_option='Fraction')

    if plot:
        for i, res in enumerate(full_res):
            if c_val != 0:
                str_c_val = str(c_val).replace('/', '_')
            else:
                str_c_val = str(c_val)
            if t_val != 0:
                str_t_val = str(t_val).replace('/', '_')
            else:
                str_t_val = str(t_val)
            if f_val != 0:
                str_f_val = str(f_val).replace('/', '_')
            else:
                str_f_val = str(f_val)
                
            folder_path = '../results/' + method + '_' + str(N_firms) + '_firms_' + str(N_locations) + '_locations_' + str_c_val + '_c_' + str_t_val + '_t_' + str_f_val + '_f_' + str(i) + '_path'

            file_path = folder_path + '/' + method + '_' + str(N_firms) + '_firms_' + str(N_locations) + '_locations_' + str_c_val + '_c_' + str_t_val + '_t_' + str_f_val + '_f_' + str(i) + '_path.tex'

            if os.path.isdir(folder_path):
                generate_plot(file_path, res)
            else:
                os.mkdir(folder_path)
                generate_plot(file_path, res)

    return full_res, full_res_2, i_dicts_list, pi_dict, price_dict

N_firms = 3
N_locations = 21
c_val = 1 # Variable cost
t_val = Fraction(1, 2) # Travel cost
f_val = 0 #Fraction(1, 50) # Fixed cost
V_val = 100 # Maximum consumer utility

inputs = N_firms, N_locations, c_val, t_val, f_val, V_val

full_res, full_res_2, i_dicts_list, pi_dict, price_dict = run_model(inputs, method='exogenous', plot=True)

"""
Author : Dhaval Kumar
Created on : 20-08-2021
Time : 09:22:43 pm 
"""

# Imports
import os.path
import sys
import numpy as np
from icecream import ic

# File Handeling
if os.path.exists('input.txt'):
    sys.stdin = open('input.txt', 'r')
    sys.stdout = open('output.txt', 'w')

# Function
def is_minq(s : list[str]) -> bool:
    """
    Given an objective function in string is of min or not.
    """
    if s[0].lower() == 'min':
        return True
    else:
        return False

def obj_fn_parser(s : list[str]) -> tuple[int, dict]:
    """
    Given an objective function in string 
    returns number of varibles in it and dict of all varibles as key and its coeff as value .
    """
    xs = s[3:]

    # dbg
    # ic(xs)

    var_coeff = {}
    for vwc in xs:
        idx_x = vwc.find('x') 
        var_coeff[vwc[idx_x:]] = float(vwc[:idx_x])
    
    # ic(var_coeff)

    return (len(var_coeff), var_coeff)

def parse_equation(eq : list[str], num_of_x : int) -> tuple[list[float], str, float]:
    """
    Parses the string list and returns a list of all varible with coefficient (row) , sign(string) and b.
    """
    
    lst = [float(0) for i in range(num_of_x)]
    b = 0
    
    for x in eq:
        # ic(x)

        if x.find('=') != -1:
            break

        idx_x = x.find('x')

        if x[:idx_x] == '+':
            lst[int(x[idx_x+1:])-1] = float(1)
        elif x[:idx_x] == '-':
            lst[int(x[idx_x+1:])-1] = float(-1)
        else :
            lst[int(x[idx_x+1:])-1] = float(x[:idx_x])
    
    b = float(eq[-1])
    
    if(b < 0):
        b *= -1
        for i in range(len(lst)):
            lst[i] *= -1

    sign = eq[-2]
    # ic(lst)
    # ic(sign)
    # ic(b)
    return (lst, sign, b)

def add_slack_overflow_artficial_var(mat : list[list[float]], signs : list[str], costs : list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] :
    """
    Convert everything to numpy.

    Returns : 3 numpy matix i.e numpy array of size=(n, m)

    Finally XD
    """
    # Don't do this but I had no other choice

    np_mat = np.array(mat)
    np_cost = np.zeros(len(costs)) # Zeros

    np_cost_p2 = np.array(costs)
    
    # VVIMP
    np_cost_idx = np.zeros(len(mat), dtype=np.int64)

    np_ibfs = np.zeros(len(mat))

    # ic(np_mat)
    # ic(np_cost)
    
    extended_len = 0

    no_of_s = 0
    no_of_a = 0

    # First sn, second an
    add_var = [ [False, False] for i in range(len(signs)) ]
    for i in range(len(signs)):
        if signs[i] == '=':
            # 1 an 
            extended_len += 1
            no_of_a += 1
            
            # For 1 an
            add_var[i][1] = True

        elif signs[i] == '<=':
            # 1 sn
            extended_len += 1
            no_of_s += 1

            # For 1 sn
            add_var[i][0] = True

        elif signs[i] == '>=':
            # 1 sn, 1 an
            extended_len += 2
            no_of_s += 1
            no_of_a += 1

            # For 1 an, 1 sn
            add_var[i][0] = True
            add_var[i][1] = True
    
    extended_mat = np.zeros((len(mat), extended_len))
    extended_cost = np.zeros(extended_len)

    for i in range(extended_len):
        extended_cost[i] = 1
    for i in range(no_of_s):
        extended_cost[i] = 0

    np_f_cost = np.hstack((np_cost, extended_cost))

    idx_s = 0
    idx_a = no_of_s

    k = 0
    for i in range(len(mat)):
        if add_var[i][0] and add_var[i][1]:
            extended_mat[i][idx_s] = -1
            extended_mat[i][idx_a] = 1
            
            np_ibfs[k] = np_f_cost[len(mat[i])+idx_a]
            np_cost_idx[k] = len(mat[i])+idx_a
            k += 1
            
            idx_a += 1
            idx_s += 1
        elif (not add_var[i][0]) and add_var[i][1]:
            extended_mat[i][idx_a] = 1

            np_ibfs[k] = np_f_cost[len(mat[i])+idx_a]
            np_cost_idx[k] = len(mat[i])+idx_a
            k += 1

            idx_a += 1
        elif add_var[i][0] and (not add_var[i][1]):
            extended_mat[i][idx_s] = 1

            np_ibfs[k] = np_f_cost[len(mat[i])+idx_s]
            np_cost_idx[k] = len(mat[i])+idx_s
            k += 1
            
            idx_s += 1

    np_f_mat = np.hstack((np_mat, extended_mat))
    # ic(np_f_mat)
    # ic(np_f_cost)
    extended_cost_p2 = np.zeros(extended_len)

    np_cost_p2 = np.hstack((np_cost_p2, extended_cost_p2))
    # ic(np_cost_p2)
    # ic(extended_mat)
    # ic(extended_cost)
    # ic(add_var)
    # ic(no_of_a)
    # ic(no_of_s)
    # ic(extended_len)

    # ic(np_f_cost)
    # ic(np_f_mat)
    # ic(np_ibfs)

    # ic(np_cost_idx)
    return (np_f_mat, np_f_cost, np_ibfs, np_cost_idx, np_cost_p2)

def calc_Cjbars(mat : np.ndarray, costs : np.ndarray, basic_var : np.ndarray) -> np.ndarray:
    """
    Calculates the Cj bar for every column
    """
    # ic(mat)
    # # ic(len(mat[0]))
    # ic(costs)
    # # ic(len(costs))
    # ic(basic_var)

    Cj = np.zeros(len(costs))
    
    for i in range(len(mat[0])):
        for j in range(len(mat)):
            Cj[i] += basic_var[j]*mat[j][i]

    Cj_bar = np.zeros(len(costs))
    for i in range(len(Cj_bar)):
        Cj_bar[i] = costs[i] - Cj[i]

    # ic(Cj_bar)

    return Cj_bar

def selected_column(Cj_bars : np.ndarray) -> int:
    """
    Returns -1 if solution is reached or infeasible || returns index of selected column 
    """
    col = -1
    cur = 1
    for i in range(len(Cj_bars)):
        if(Cj_bars[i] < 0 and Cj_bars[i] < cur):
            cur = Cj_bars[i]
            col = i
    return col

def selected_row(mat : np.ndarray, bs : np.ndarray, col : int) -> int:
    inf = float(999999)
    ratio = np.zeros(len(bs))
    for i in range(len(mat)):
        if mat[i][col] <= 0:
            ratio[i] = inf
        else:
            ratio[i] = bs[i]/mat[i][col]
    
    # ic(ratio)

    row = -1
    cur = inf-1
    for i in range(len(ratio)):
        if (ratio[i] < cur):
            cur = ratio[i]
            row = i
    
    # ic(row)

    return row

def change_of_bv(row : int, col : int, basic_var : np.ndarray, costs : np.ndarray, cost_idx) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns basic varibles, cost idx
    
    IMP : cost idx array is very important for calculating the solution
    """
    
    basic_var[row] = costs[col]
    cost_idx[row] = col
    # ic(basic_var)
    # ic(cost_idx)
    return (basic_var, cost_idx) 


# NOTE : include bs in this funtion really imp
def gauss_jorden(row : int, col : int, mat : np.ndarray, bs : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Does all the row wise operation that are required and returns new mat and bs
    """
    # ic(mat)
    # ic(bs)
    bs[row] = bs[row]/mat[row][col]
    mat[row] = mat[row]/mat[row][col] 
    
    for i in range(len(mat)):
        if row == i:
            continue
        else:
            bs[i] = bs[row]*(-mat[i][col])+ bs[i]
            mat[i] = mat[row]*(-mat[i][col]) + mat[i]

    # ic(mat)
    # ic(bs)

    return (mat, bs)

def calc_soln(costs_idx : np.ndarray, costs : np.ndarray, bs : np.ndarray) -> float:
    """
    Returns the final Z 
    Finally XD
    """
    Z = float(0)
    for i in range(len(costs_idx)):
        Z += costs[costs_idx[i]]*bs[i]
    
    return Z

def reshape(mat : np.ndarray, prv_cost : np.ndarray) -> np.ndarray:
    for i in range(len(mat)):

        for j in range(len(mat[i])):
            if prv_cost[j] == 1:
                mat[i][j] = 0

    return np.array(mat)
# Main 
if __name__ == "__main__":
    # Initializing variables
    minq = False

    # Input
    no_of_eq = int(input())
    Z = input().split()
    
    # Useful
    minq = is_minq(Z)
    obj_num_of_x, obj_coeff_of_x = obj_fn_parser(Z)
    costs = [i for i in obj_coeff_of_x.values()] 
    
    # For max problem 
    if (not minq):
        costs = [-i for i in obj_coeff_of_x.values()] 


    # ic(obj_coeff_of_x)
    # ic(obj_num_of_x)

    eq = [] # for input
    
    # Useful
    signs = ['' for i in range(no_of_eq)]
    mat = [[] for i in range(no_of_eq)]
    bs = [float(0) for i in range(no_of_eq)]
    
    for i in range(no_of_eq):
        eq = input().split()
        mat[i], signs[i], bs[i] = parse_equation(eq, obj_num_of_x)

    # ic(mat)
    # ic(signs)
    # ic(bs)
    # ic(costs)

    # NumPy everything from here

    # Phase - I

    np_bs = np.array(bs)
    np_mat , np_costs, np_basic_var, np_costs_idx, np_costs_p2 = add_slack_overflow_artficial_var(mat, signs, costs)

    # NOTE : change np_cost here for maximization problem DOESNT WORK
    # if (not minq):
    #     print("HERE")
    #     print(np_costs)
    #     np_costs = -1*np_costs
    #     print(np_costs)
        
    Cjbars = calc_Cjbars(np_mat, np_costs, np_basic_var)
    
    col = selected_column(Cjbars)
    row = selected_row(np_mat, np_bs, col)
    
    
    while(col != -1 and row != -1):
        np_basic_var, np_costs_idx = change_of_bv(row, col, np_basic_var, np_costs, np_costs_idx)
        
        np_mat, np_bs = gauss_jorden(row, col, np_mat, np_bs) 

        # Cycle starts
        Cjbars = calc_Cjbars(np_mat, np_costs, np_basic_var)
    
        col = selected_column(Cjbars)
        row = selected_row(np_mat, np_bs, col)


    # Phase - II

    np_mat = reshape(np_mat, np_costs)

    Cjbars = calc_Cjbars(np_mat, np_costs_p2, np_basic_var)
    
    col = selected_column(Cjbars)
    row = selected_row(np_mat, np_bs, col)
    
    # Phase - I
    while(col != -1 and row != -1):
        np_basic_var, np_costs_idx = change_of_bv(row, col, np_basic_var, np_costs_p2, np_costs_idx)
        
        np_mat, np_bs = gauss_jorden(row, col, np_mat, np_bs) 

        # Cycle starts
        Cjbars = calc_Cjbars(np_mat, np_costs_p2, np_basic_var)
    
        col = selected_column(Cjbars)
        row = selected_row(np_mat, np_bs, col)
        # ic(np_mat)

    if (not minq):
        # print("HERE")
        # print(np_costs)
        np_costs_p2 = -1*np_costs_p2
        # print(np_costs)

    # ic(np_mat)

    print(f"Z = {calc_soln(np_costs_idx, np_costs_p2, np_bs)}")
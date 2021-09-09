"""
Author : Dhaval Kumar
Created on : 23-08-2021
Time : 11:22:43 am 
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

# Functions

def findRow(mat : np.ndarray) -> int:
    row = -1
    cur = 0
    for i in range(len(mat)):
        if mat[i][-1] < cur:
            row = i
            cur = mat[i][-1]
    return row

def calcCjbar(costs : np.ndarray, row : int, mat : np.ndarray, basic_var : np.ndarray) -> np.ndarray:
    Cj = np.zeros(len(costs), dtype=np.float64)

    for i in range(len(costs)):
        for j in range(len(mat)):
            Cj[i] += mat[j][i]*basic_var[j]
    
    for i in range(len(costs)):
        Cj[i] = costs[i] - Cj[i]
    
    # ic(costs)
    # ic(basic_var)
    # ic(mat)
    # ic(Cj)

    return Cj

def findCol(mat : np.ndarray, Cj_bar : np.ndarray, row : int) -> int:
    # col = -1
    ratio = np.zeros(len(Cj_bar), dtype=np.float64)
    for i in range(len(Cj_bar)):
        if mat[row][i] < 0:
            ratio[i] = abs(Cj_bar[i]/mat[row][i])
        else:
            ratio[i] = float(99999)
    # ic(ratio)

    col = -1
    cur = 99999
    for i in range(len(ratio)):
        if ratio[i] < cur:
            col = i
            cur = ratio[i]
    # ic(Cj_bar)
    # ic(col)
    return col

def changeBV(row : int, col : int, costs : np.ndarray, idx_of_bv : np.ndarray, basic_var : np.ndarray) -> list[np.ndarray, np.ndarray]:
    basic_var[row] = costs[col]
    idx_of_bv[row] = col

    return (basic_var, idx_of_bv)

def gauss_jorden(row : int, col : int, mat : np.ndarray) -> np.ndarray:
    mat[row] = mat[row]/mat[row][col]
    for i in range(len(mat)):
        if row == i:
            continue
        else:
            mat[i] = mat[row]*(-mat[i][col]) + mat[i]
    return mat

def calcSoln(idx_of_bv : np.ndarray, costs : np.ndarray, mat : np.ndarray) -> float:
    Z = float(0)
    ic(costs)
    ic(mat)
    for i in range(len(idx_of_bv)):
        Z += costs[idx_of_bv[i]]*(mat[i][-1])
        ic(mat[i][-1])
    return Z

# Main
if __name__ == '__main__':

    # Debug mode off
    # ic.disable()

    print("Minimization Problem Solver Using Dual Simplex Method :")

    no_of_lte = int(input("Number of eq with <= : "))
    no_of_gte = int(input("Number of eq with >= : "))
    no_of_eq = int(input("Number of eq with = : "))

    no_of_costs = int(input("Number of cost var : "))
    
    costs = [0 for i in range(no_of_costs)]

    print("Type the coeff of var in objective function : ")
    costs = list(map(float, input().split()))
    prev_cost_len = len(costs)

    mat = []
    temp = []

    print("Type the coeff of eq with <= : ")
    for i in range(no_of_lte):
        temp = list(map(float, input().split()))
        mat.append(temp)
        costs.append(float(0))
        
    print("Type the coeff of eq with >= : ")
    for i in range(no_of_gte):
        temp = list(map(float, input().split()))
        mat.append([-temp[i] for i in range(len(temp))])
        costs.append(float(0))
    
    print("Type the coeff of eq with = : ")
    for i in range(no_of_eq):
        temp = list(map(float, input().split()))
        mat.append(temp)
        mat.append([-temp[i] for i in range(len(temp))])
        costs.append(float(0))
        costs.append(float(0))


    len_of_extended_mat = len(costs)-(len(mat[0])-1)
    idx_of_bv = [0 for i in range(len(mat))]
    basic_var = [0 for i in range(len(mat))]

    for i in range(prev_cost_len,len(costs)):
        idx_of_bv[i-prev_cost_len] = i 
    

    extended_mat = np.identity(len_of_extended_mat, dtype=np.float64)
    # ic(extended_mat)
    # ic(mat)
    bs = []
    for i in range(len(mat)):
        bs.append([mat[i][-1]])
        del mat[i][-1]

    # ic(bs)
    ic(mat)

    # NumPy Everything

    mat = np.array(mat, dtype=np.float64)
    costs = np.array(costs, dtype=np.float64)
    mat = np.hstack((mat, extended_mat))
    mat = np.hstack((mat, np.array(bs, dtype=np.float64)))
    idx_of_bv = np.array(idx_of_bv, dtype=np.int64)
    basic_var = np.array(basic_var, dtype=np.float64)
    # ic(mat)

    # ic(mat)
    # ic(costs)
    # ic(idx_of_bv)
    # ic(basic_var)
    ic(mat)
    ic(basic_var)
    ic(costs)
    ic(idx_of_bv)
    row = findRow(mat)
    Cj_bar = calcCjbar(costs, row, mat, basic_var)
    col = findCol(mat, Cj_bar, row)

    while(True):
        basic_var , idx_of_bv = changeBV(row, col, costs, idx_of_bv, basic_var)

        mat = gauss_jorden(row, col, mat)

        ic(mat)
        ic(basic_var)
        ic(costs)
        ic(idx_of_bv)

        # Cycle Starts
        row = findRow(mat)
        if (row == -1):
            break
        Cj_bar = calcCjbar(costs, row, mat, basic_var)
        col = findCol(mat, Cj_bar, row)
        if (col == -1):
            break

    print("Output : ")
    print(f"Z = {calcSoln(idx_of_bv, costs, mat)}")



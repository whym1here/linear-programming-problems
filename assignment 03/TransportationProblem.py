"""
Author : Dhaval Kumar
Created on : 23-08-2021
Time : 6:20:33 pm 
"""

# Imports
import os.path
import sys
import numpy as np
from icecream import ic

# File Handling
# if os.path.exists('input.txt'):
#     sys.stdin = open('input.txt', 'r')
#     sys.stdout = open('output.txt', 'w')

# Functions
def NorthWest(cost_mat : np.ndarray, rows : int, cols : int, supply : np.ndarray, demand : np.ndarray) -> np.ndarray:
    soln = np.zeros((rows, cols))
    i, j = 0, 0

    # Using Two Pointer Method O(n+m-1) 
    for _ in range(rows+cols-1):
        # Going down a row
        if (supply[i] < demand[j]):
            soln[i, j] = supply[i]
            demand[j] -= supply[i]
            supply[i] = 0
            i += 1

        # Going down a column and a row
        elif (demand[j] == supply[i]):
            soln[i, j] = demand[j]
            demand[j] = 0
            supply[i] = 0
            i += 1
            j += 1

        # Going down a column
        else:
            soln[i, j] = demand[j]
            supply[i] -= demand[j]
            demand[j] = 0
            j += 1

    return soln

def find_least_cell(cost_mat : np.ndarray, check : np.ndarray) -> tuple[int, int]:
    val = 99999
    row = -1
    col = -1
    for i in range(len(cost_mat)):
        for j in range(len(cost_mat[i])):
            if (check[i][j] == False and cost_mat[i][j] < val):
                row, col = i, j
                val = cost_mat[i][j]
    # ic(row)
    # ic(col)
    # ic(cost_mat)
    return (row, col)


def LeastCost(cost_mat : np.ndarray, rows : int, cols : int, supply : np.ndarray, demand : np.ndarray) -> np.ndarray:
    check = np.zeros((rows, cols), dtype=bool)
    soln = np.zeros((rows, cols), dtype=np.float64)
    row, col = find_least_cell(cost_mat, check)
    while(row != -1 and col != -1):
        # ic(check)
        # ic(soln)
        if(supply[row] > demand[col]):
            soln[row][col] = demand[col]

            supply[row] -= demand[col]
            demand[col] = 0
            check[:, col] = True

        elif(supply[row] == demand[col]):
            soln[row][col] = demand[col]

            supply[row] = 0
            demand[col] = 0
            check[:, col] = True
            check[row, :] = True
        
        else:
            soln[row][col] = supply[row]

            demand[col] -= supply[row]
            supply[row] = 0
            check[row, :] = True
        
        row, col = find_least_cell(cost_mat, check)
    
    return soln

def max_penalty_index(mat : np.ndarray, r : int, c : int) -> tuple[int, int]:
    row_penalties, col_penalties = np.array([0] * r), np.array([0] * c)
    for i in range(r):
        row = np.sort(mat[i])
        row_penalties[i] = row[1] - row[0]
        # ic(row)
        # ic(row_penalties)
    mat = mat.transpose()
    for i in range(c):
        col = np.sort(mat[i])
        col_penalties[i] = col[1] - col[0]
        # ic(col)
        # ic(col_penalties)
    row_max = np.argmax(row_penalties)
    col_max = np.argmax(col_penalties)

    # ic(row_max)
    # ic(col_max)

    if row_penalties[row_max] > col_penalties[col_max]:
        return (0, row_max)
    else:
        return (1, col_max)

def vogel_approx(cost_mat : np.ndarray, r : int, c : int, supply : np.ndarray, demand : np.ndarray):
    soln = np.zeros((r, c))
    basics_var = np.array([[np.nan, np.nan]]*(r+c-1))
    cnt = 0
    cost_replacement = np.amax(cost_mat) + 1

    while (cnt < r+c-1):
        # ic(max_penalty)
        # ic(max_penalty_index)
        # ic(supply)
        # ic(demand)
        # ic(soln)
        # ic(cnt)

        max_penalty = max_penalty_index(cost_mat, r, c)
        if max_penalty[0] == 0:
            least_cost_col_index = np.argmin(cost_mat[max_penalty[1]])
            if supply[max_penalty[1]] > demand[least_cost_col_index]:
                soln[max_penalty[1], least_cost_col_index] = demand[least_cost_col_index]
                supply[max_penalty[1]] -= demand[least_cost_col_index]
                demand[least_cost_col_index] = 0

                cost_mat[:, least_cost_col_index] = cost_replacement
            
            elif supply[max_penalty[1]] == demand[least_cost_col_index]:
                soln[max_penalty[1], least_cost_col_index] = demand[least_cost_col_index]
                supply[max_penalty[1]] = 0
                demand[least_cost_col_index] = 0

                cost_mat[max_penalty[1]] = cost_replacement
                cost_mat[:, least_cost_col_index] = cost_replacement
            else:
                soln[max_penalty[1], least_cost_col_index] = supply[max_penalty[1]]

                demand[least_cost_col_index] -= supply[max_penalty[1]]
                supply[max_penalty[1]] = 0

                cost_mat[max_penalty[1]] = cost_replacement

            basics_var[cnt] = [max_penalty[1], least_cost_col_index]

        else:
            least_cost_row_index = np.argmin(cost_mat[:,max_penalty[1]])

            if demand[max_penalty[1]] > supply[least_cost_row_index]:
                soln[least_cost_row_index, max_penalty[1]] = supply[least_cost_row_index]

                demand[max_penalty[1]] -= supply[least_cost_row_index]
                supply[least_cost_row_index] = 0

                cost_mat[least_cost_row_index] = cost_replacement
            
            elif demand[max_penalty[1]] == supply[least_cost_row_index]:
                soln[least_cost_row_index, max_penalty[1]] = demand[max_penalty[1]]
                supply[least_cost_row_index] = 0
                demand[max_penalty[1]] = 0
                
                cost_mat[:,max_penalty[1]] = cost_replacement
                cost_mat[least_cost_row_index] = cost_replacement

            else:
                soln[least_cost_row_index, max_penalty[1]] = demand[max_penalty[1]]
                supply[least_cost_row_index] -= demand[max_penalty[1]]
                demand[max_penalty[1]] = 0

                cost_mat[:,max_penalty[1]] = cost_replacement
            basics_var[cnt] = [least_cost_row_index, max_penalty[1]]
        cnt += 1

    return soln

# Main
if __name__=='__main__':
    # Debug Mode off
    # ic.disable()
    print("1) North-West")
    print("2) Least Cost")
    print("3) Vogel Approx")
    print("CHOOSE which method to find IBFS : ")
    choice = int(input())

    
    cols = int(input("No. of cost cols : "))
    rows = int(input("No. of cost rows : "))
    
    print("Enter supply array here : ")
    supply = list(map(float, input().split()))

    print("Enter demand array here : ")
    demand = list(map(float, input().split()))

    cost_mat = [ [] for i in range(rows) ]
    print("Enter the cost table here : ")
    for i in range(rows):
        cost_mat[i] = list(map(float, input().split()))

    # NumPy everything
    supply = np.array(supply)
    demand = np.array(demand)
    cost_mat = np.array(cost_mat)
    # sol = np.zeros((rows, cols), dtype=np.float64)
    # ic(sol)
    
    # ic(supply)
    # ic(demand)
    # ic(cost_mat)
    # ic(NorthWest(cost_mat, rows, cols, supply, demand))

    # ic(LeastCost(cost_mat, rows, cols, supply, demand))


    print("IBFS : ")
    if choice == 1:
        print("Using North-West :")
        print(NorthWest(cost_mat, rows, cols, supply, demand))
    elif choice == 2:
        print("Using Least Cost :")
        print(LeastCost(cost_mat, rows, cols, supply, demand))
    elif choice == 3:
        print("Using Vogel's Approximation :")
        print(vogel_approx(cost_mat, rows, cols, supply, demand))
    else:
        print("Invalid Choice")
    
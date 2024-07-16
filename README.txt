### Author Florian Roesel 08.09.2023

To run the code, we recommend
- python 3.10.12
- gurobi 10.0.2
- gurobipy 10.0.2

The program is dedicated to compare cut selection strategies for Benders Decomposition.
The program takes a mixed-integer linear program in appropriate file format (.lp or .mps)
as input and solves it with Benders Decomposition.

# First: PREPARE an instance; type

python3 instance_prep.py -i <INPUT_FOLDER> -o <OUTPUT_FOLDER> <INSTANCE_NAME>
e.g.,
python3 instance_prep.py -i Persistent_Random -o Prep_Random random_100_200_100_100_200

to take an arbitrary MIP as .lp file or .mps file as input and to create
    1. an .lp file that contains the Benders Master Problem (containing only the integer variables of the original MIP)
    2. an .lp file that contains the Dual Benders Subproblem
    3. the constraint matrix of constraints that contain integer and continuous variables, of the integer variables
    4. a solution of the original problem as .sol file
    5. an approximate (25%) solution of the original problem, as .sol file
    6. some auxiliary files
    
These files are compatible with compare_css.py, which reads instances in the described format.
(type "python3 instance_prep.py --help" for further information)

# Then, Solve the prepared instance; type

python3 compare_css.py --instancepath <INPUT_FOLDER> -m <MODE> <INSTANCE_NAME>
e.g.,
python3 compare_css.py --instancepath Prep_Random -m 0 random_100_200_100_100_200

to solve a prepared instance. The modes can be chosen from
0: Classical Benders
1: Magnanti-Wong
2: MIS Benders
3: Facet Cuts
4: Optimal Line-Shifting Cuts
5: MIP Solver
6: Hybrid
7: The CPLEX Recipe following Bonami et al. (2020)
8: The CPLEX Recipe using OLS cuts instead of feasibility cuts

(type "python3 compare_css.py --help" for further information)

## Furthermore...
The program "generate_grid.py" generates NWD instances. Type "python3 generate_grid.py --help" for further information.
The program "create_MIP.py" generates Random MIP instances. Type "python3 create_MIP.py --help" for further information.

### Questions? Contact florian.roesel@fau.de / florian.roesel@outlook.com


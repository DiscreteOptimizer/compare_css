### Author Florian Roesel 08.09.2023

import gurobipy as grb
import argparse
import logging
logger = logging.getLogger("__main__")
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
import random
import json
import networkx

def indicator_random(n):
    """
    Returns 1 with probability 1/n and 0 with probability (n-1)/n
    """
    return int(np.random.randint(n) == 0)

def generate_basic_MIP(instance_name, var_master = 10, var_sub = 20, cons_master = 10, cons_mix = 10, cons_sub = 20, instance_path = "Instances"):
    
    """
    This method generates a random MIP - simply sampling the coefficients of the master
    matrix, the coupling matrix and the dualsub matrix.
    Input:
    var_master: How many integer variables?
    var_sub: How many continuous variables?
    cons_master: How many constraints containing only integer variables?
    cons_mix: How many constraints containing both integer and continuous variables?
    cons_sub: How many constraints containing only continuous variables?
    """
    
    if os.path.exists(f"{instance_path}"):
        pass
    else:
        os.mkdir(f"{instance_path}")
    
    output_path = f"{instance_path}/{instance_name}.lp"
    
    overall_prob = grb.Model("Random Model")
    
    # fields for master variables
    master_vars = []
    sub_vars = []
    
    # "densities" for integer/continuous nonzero coefficients - currently hard coded
    master_density = int(0.1*var_master) + 1
    sub_density = int(0.05*var_sub) + 1
    
    # add the variables
    for i in range(var_master):
        master_vars.append(overall_prob.addVar(vtype = "I", obj = 0.5 * (np.random.randint(10) + 1), name = f"master_var_{i}"))
    for i in range(var_sub):
        sub_vars.append(overall_prob.addVar(vtype = "C", obj = var_master / var_sub * (np.random.randint(10) + 0.01 * np.random.randint(100)), name = f"sub_var_{i}"))
    overall_prob.update()
    
    # fields for constraints
    master_cons = []
    mix_cons = []
    sub_cons = []
    
    # master constraints
    print("Adding master constraints.")
    for i in range(cons_master):
        print(f"{i}/{cons_master}", end="\r")
        master_cons.append(overall_prob.addConstr(sum(indicator_random(master_density) * np.random.randint(10) * var for var in master_vars) >= .1 * np.random.randint(40) * var_master, name = f"master_cons_{i}"))
    print("Master constraints added.")
    
    # mix constraints
    # a couple of equalities
    for i in range(5):
        mix_cons.append(overall_prob.addConstr(sum(indicator_random(master_density) * np.random.randint(10) * var for var in master_vars) + sum(indicator_random(sub_density) * np.random.randint(10) * var for var in sub_vars) == 1. + np.random.randint(10) * (var_master + var_sub), name = f"eq_cons_{i}"))
    
    # a couple of inequalities
    print("Adding mix constraints.")
    for i in range(cons_mix):
        print(f"{i}/{cons_mix}", end="\r")
        mix_cons.append(overall_prob.addConstr(sum(indicator_random(master_density) * np.random.randint(10) * var for var in master_vars) + sum(indicator_random(sub_density) * np.random.randint(10) * var for var in sub_vars) >= .1 * np.random.randint(40) * (var_master + var_sub), name = f"mix_cons_{i}"))
    print("Mix constraints added.")
    
    # pure sub constraints
    print("Adding pure sub constraints.")
    for i in range(cons_sub):
        print(f"{i}/{cons_sub}", end="\r")
        sub_cons.append(overall_prob.addConstr(sum(indicator_random(sub_density) * np.random.randint(10) * var for var in sub_vars) >= .1 * np.random.randint(40) * var_master, name = f"sub_cons_{i}"))
    print("Pure sub constraints added.")
    overall_prob.update()
    overall_prob.Params.MIPGap = 0.5
    overall_prob.optimize()
    if overall_prob.status != 2:
        # repeat if something infeasible comes out
        generate_basic_MIP(instance_name, var_master, var_sub, cons_master, cons_mix, cons_sub, instance_path)
        return
    overall_prob.write(output_path)
    logger.message("Instance created.")

if __name__ == "__main__":
    """
    The main script.
    """
    parser = argparse.ArgumentParser(description='Cut selection strategies - create random MIPs for the computational tests.')
    parser.add_argument("instance")
    parser.add_argument("-i", "--instance_path",
                        help="Which folder to put the instance to.",
                        default = "Random_Instances")
    parser.add_argument("-c", "--config",
                        default = "10_20_10_10_20")
    
    # https://stackoverflow.com/questions/14097061/easier-way-to-enable-verbose-logging
    parser.add_argument(
        '-d', '--debug',
        help="Print lots of debugging statements",
        action="store_const", dest="loglevel", const=20,
        default=logging.WARNING,
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=25,
        default=logging.WARNING,
    )
    parser.add_argument(
        '-s', '--random_seed',
        help="Random seed.",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()
    random.seed(args.random_seed)
    logging.basicConfig(level=args.loglevel,
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        handlers=[logging.FileHandler("debug.log"),
                                  logging.StreamHandler()])
    logging.MESSAGE_LEVEL = 25
    logging.addLevelName(logging.MESSAGE_LEVEL, "MSGE")
    logging.message=lambda mess, *args, **kwargs: logging.log(logging.MESSAGE_LEVEL,
                                                             mess,
                                                             *args,
                                                             **kwargs)
    logger = logging.getLogger("__main__")
    logger.message=lambda mess, *args, **kwargs: logging.log(logging.MESSAGE_LEVEL,
                                                             mess,
                                                             *args,
                                                             **kwargs)
    logger.message("Verbose statements activated.")
    logger.info("Debug statements activated.")
    logger.debug("This level is not in use.")
    instance_name = args.instance
    instance_path = args.instance_path
    config = [int(element) for element in args.config.split("_")]
    var_master = config[0]
    var_sub = config[1]
    cons_master = config[2]
    cons_mix = config[3]
    cons_sub = config[4]
    
    generate_basic_MIP(instance_name, var_master, var_sub, cons_master, cons_mix, cons_sub, instance_path)


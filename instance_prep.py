### Author Florian Roesel 08.09.2023

import gurobipy as grb
import argparse
import logging
logger = logging.getLogger("__main__")
import numpy as np
import os
import sys
import random
import json


def transform_instance(instance_name, inputpath="Instances", outputpath="PrepInstances"):
    logger.message("Read Instance")
    master_vars = {}
    sub_vars = {}

    master_cons = {}
    mix_cons = {}
    sub_cons = {}
    all_sub_cons = {}

    # Initialize Matrix H; one entry for each mix_constr, each entry is a |master_var|-length numpy array
    H = {} 
    if os.path.exists(f"{inputpath}/{instance_name}.lp"): m = grb.read(f"{inputpath}/{instance_name}.lp")
    elif os.path.exists(f"{inputpath}/{instance_name}.mps"): m = grb.read(f"{inputpath}/{instance_name}.mps")
    else:
        logger.warning(f"Instance {instance_name} does not exist. Terminating.")
        sys.exit()
    m.update()
    for variable in m.getVars():
        if variable.vtype == "B":
            variable.vtype = "I"
            variable.lb = 0.
            variable.ub = 1.
    if os.path.exists(f"{outputpath}"):
        pass
    else:
        os.mkdir(f"{outputpath}")
    
    if os.path.exists(f"{outputpath}/{instance_name}"):
        logger.warning(f"Instance {instance_name} already prepared. Clearing.")
        os.system(f"rm -r {outputpath}/{instance_name}")
        os.mkdir(f"{outputpath}/{instance_name}")
    else:
        os.mkdir(f"{outputpath}/{instance_name}")
    
    # set MIPGap to 25% and solve to determine first incumbent
    m.Params.MIPGap = 0.25
    m.update()
    m.optimize()
    if m.status != 2:
        m.computeIIS()
        m.write("ex.ilp")
        os.system(f"rm -r {outputpath}/{instance_name}")
        system.warning("Instance {instance_name} cannot be solved to optimality. Terminating and deleting {outputpath}/{instance_name} directory.")
        sys.exit()
    m.write(f"{outputpath}/{instance_name}/master_0.25.sol")
    m.write(f"{outputpath}/{instance_name}/{instance_name}.lp")
    m.write(f"{outputpath}/{instance_name}/optimal.sol")
    
    master = grb.Model(f"{instance_name}_master")
    core_dict = {}
    print("Going through variables")
    count = 0
    var_length = len(m.getVars())
    steps = int(var_length/80) + 1
    for variable in m.getVars():
        count += 1
        if count % steps == 0:
            print("-", end = "", flush=True)
        if variable.vtype == "I" or variable.vtype == "B":
            master_vars[variable.varname] = variable
            core_dict[variable.varname] = round(variable.X)
        else:
            sub_vars[variable.varname] = variable
            if variable.lb not in [- grb.GRB.INFINITY, 0., -float("inf")]:
                m.addConstr(variable >= variable.lb, name = variable.varname + "_lb")
                if variable.lb > 0:
                    variable.lb = 0
                else:
                    variable.lb = - grb.GRB.INFINITY
            if variable.ub not in [grb.GRB.INFINITY, 0., float("inf")]:
                m.addConstr(variable <= variable.ub, name = variable.varname + "_ub")
                if variable.ub > 0:
                    variable.ub = grb.GRB.INFINITY
                else:
                    variable.ub = 0
    
    with open(f"{outputpath}/{instance_name}/core.json", "w") as my_file:
        json.dump(core_dict, my_file)
    
    m.update()
    # define f
    f = np.array([var.obj for _, var in master_vars.items()])
    logger.info(len(m.getVars()), len(master_vars), len(sub_vars))
    print("")
    print("Going through constraints")
    count = 0
    var_length = len(m.getConstrs())
    steps = int(var_length/80) + 1
    
    for cons in m.getConstrs():
        count += 1
        if count % steps == 0:
            print("-", end = "", flush=True)
        if cons.constrname.endswith("_lb") or cons.constrname.endswith("_ub"):
            sub_cons[cons.constrname] = cons
            continue
        at_least_one_master = False
        at_least_one_sub = False
        for vname, variable in master_vars.items():
            if m.getCoeff(cons, variable) != 0.:
                at_least_one_master = True
                break
        if at_least_one_master:
            for vname, variable in sub_vars.items():
                if m.getCoeff(cons, variable) != 0.:
                    at_least_one_sub = True
                    break
        else:
            at_least_one_sub = True
        
        # create H-Matrix
        if at_least_one_master and at_least_one_sub:
            mix_cons[cons.constrname] = cons
            tmp = []
            for vname, variable in master_vars.items():
                tmp.append(m.getCoeff(cons, variable))
            H[cons.constrname] = tmp
        if at_least_one_master and not at_least_one_sub:
            master_cons[cons.constrname] = cons
        if at_least_one_sub and not at_least_one_master:
            sub_cons[cons.constrname] = cons
    order = []
    for vname in master_vars:
        order.append(vname)
    
    # dump H to json file
    with open(f"{outputpath}/{instance_name}/H.json", "w") as my_file:
        json.dump(H, my_file)
    with open(f"{outputpath}/{instance_name}/order.json", "w") as my_file:
        json.dump(order, my_file)
    
    all_sub_cons.update(mix_cons)
    all_sub_cons.update(sub_cons)
    
    logger.info(f"{len(m.getConstrs())}, {len(master_cons)}, {len(mix_cons)}, {len(sub_cons)}, {len(all_sub_cons)}")
    
    # build master problem
    master = grb.Model("Master")
    
    master._vars = {}
    master._aux = master.addVar(obj = 1., lb = 0., name = "aux")
    
    print("")
    print("Creating master")
    count = 0
    var_length = len(master_vars) + len(master_cons)
    steps = int(var_length/80) + 1
    
    for m_var_name, m_var in master_vars.items():
        count += 1
        if count % steps == 0:
            print("-", end = "", flush=True)
        master._vars[m_var_name] = master.addVar(obj = m_var.obj, lb = m_var.lb, ub = m_var.ub, name=m_var_name, vtype = m_var.vtype)
    
    master.update()

    # Set up pure Master Constraints.
    for master_constr_name, master_constr in master_cons.items():
        count += 1
        if count % steps == 0:
            print("-", end = "", flush=True)
        if master_constr.Sense == "<":
            master.addConstr(sum(m.getCoeff(master_constr, var) * master._vars[var_name] for var_name, var in master_vars.items()) <= master_constr.RHS, name = master_constr_name)
        elif master_constr.Sense == "=":
            master.addConstr(sum(m.getCoeff(master_constr, var) * master._vars[var_name] for var_name, var in master_vars.items()) == master_constr.RHS, name = master_constr_name)
        elif master_constr.Sense == ">":
            master.addConstr(sum(m.getCoeff(master_constr, var) * master._vars[var_name] for var_name, var in master_vars.items()) >= master_constr.RHS, name = master_constr_name)
        else:
            logger.warning("Master constraint adding error. Terminating.")
            sys.exit()
    master.update()
    
    dualsub = grb.Model("DualSub")
    dualsub.ModelSense = grb.GRB.MAXIMIZE
    dualsub._vars = {}
    dualsub._constrs = {}
    print("")
    print("Creating sub")
    count = 0
    var_length = len(sub_cons) + len(mix_cons) + len(sub_vars)
    steps = int(var_length/80) + 1
    
    # Set up pure Dual Constraints.
    for sub_constr_name, sub_constr in sub_cons.items():
        count += 1
        if count % steps == 0:
            print("-", end = "", flush=True)
        if sub_constr.Sense == "<":
            lb = - grb.GRB.INFINITY
            ub = 0.
        elif sub_constr.Sense == "=":
            lb = - grb.GRB.INFINITY
            ub = grb.GRB.INFINITY
        elif sub_constr.Sense == ">":
            lb = 0.
            ub = grb.GRB.INFINITY
        else:
            logger.warning("Invalid constraint sense. Terminating.")
            sys.exit()
        dualsub._vars[sub_constr_name] = dualsub.addVar(lb = lb, ub = ub, obj = sub_constr.RHS, name = sub_constr_name)

    # set up mixed constraints
    for mix_constr_name, mix_constr in mix_cons.items():
        count += 1
        if count % steps == 0:
            print("-", end = "", flush=True)
        
        if mix_constr.Sense == "<":
            lb = - grb.GRB.INFINITY
            ub = 0.
            coeff = -1.
        elif mix_constr.Sense == "=":
            lb = - grb.GRB.INFINITY
            ub = grb.GRB.INFINITY
        elif mix_constr.Sense == ">":
            lb = 0.
            ub = grb.GRB.INFINITY
        else:
            logger.warning("Invalid constraint sense. Terminating.")
            sys.exit()
        dualsub._vars[mix_constr_name] = dualsub.addVar(lb = lb, ub = ub, obj = mix_constr.RHS, name = mix_constr_name)
    
    dualsub._vars["zero"] = dualsub.addVar(lb=1., ub=1., name="zero")

    dualsub.update()

    for sub_var_name, sub_var in sub_vars.items():
        count += 1
        if count % steps == 0:
            print("-", end = "", flush=True)
        # <= constraints in the dual if lb = 0
        if sub_var.lb == 0:
            dualsub._constrs[sub_var_name] = dualsub.addConstr(sum(m.getCoeff(tmp_cons, sub_var) * dualsub._vars[tmp_name] for tmp_name, tmp_cons in all_sub_cons.items()) <= sub_var.obj * dualsub._vars["zero"], name = sub_var_name)
        # >= constraints in the dual if ub = 0
        elif sub_var.ub == 0:
            dualsub._constrs[sub_var_name] = dualsub.addConstr(sum(m.getCoeff(tmp_cons, sub_var) * dualsub._vars[tmp_name] for tmp_name, tmp_cons in all_sub_cons.items()) >= sub_var.obj * dualsub._vars["zero"], name = sub_var_name)
        # == constraints if none of the above applies
        else:
            dualsub._constrs[sub_var_name] = dualsub.addConstr(sum(m.getCoeff(tmp_cons, sub_var) * dualsub._vars[tmp_name] for tmp_name, tmp_cons in all_sub_cons.items()) == sub_var.obj * dualsub._vars["zero"], name = sub_var_name)
    
    dualsub.update()
    
    # determine lower bount for auxiliary variable..
    for variable in m.getVars():
        if variable.vtype != "C": 
            variable.obj = 0.
    m.update()
    m_relax = m.relax()
    m_relax.optimize()
    if m_relax.status == 2:
        if m_relax.objval < 0.:
            print(instance_name)
        master._aux.lb = min(m_relax.objval, 0.)
        print(master._aux.lb)
    else:
        print("Warning - relaxed problem is unbounded...")
    # write everything out
    master.update()
    master.write(f"{outputpath}/{instance_name}/master.lp")
    dualsub.write(f"{outputpath}/{instance_name}/dualsub.lp")
    print("")


if __name__ == "__main__":
    """
    Main script. Takes an arbitrary MIP as .lp file or .mps file, and creates
    1. an .lp file that contains the Benders Master Problem (containing only the integer variables of the original MIP)
    2. an .lp file that contains the Dual Benders Subproblem
    3. the constraint matrix of constraints that contain integer and continuous variables, of the integer variables
    4. a solution of the original problem as .sol file
    5. an approximate (25%) solution of the original problem, as .sol file
    6. different auxiliary files
    
    These files are compatible with compare_css.py, which reads instances in the described format.
    """
    parser = argparse.ArgumentParser(description='Instance preparation for cut selection strategies - comparison.')
    parser.add_argument("instance")
    # https://stackoverflow.com/questions/14097061/easier-way-to-enable-verbose-logging
    parser.add_argument("-i", "--inputpath",
                        help="Inputpath. Default 'Instances'",
                        default="Instances",
    )
    parser.add_argument("-o", "--outputpath",
                        help="Outputpath. Default 'PrepInstances'",
                        default="PrepInstances",
    )
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
    logger.debug("This level is not in use. Who ever wants to write a very thorough debugging, feel free.")
    instance_name = args.instance
    transform_instance(instance_name, args.inputpath, args.outputpath)


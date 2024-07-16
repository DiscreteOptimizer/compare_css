### Author Florian Roesel 08.09.2023

import gurobipy as grb
import argparse
import logging
logger = logging.getLogger("__main__")
import numpy as np
import os
import sys
import time
import random
import json
import copy as cp
from auxiliary import set_obs_attrs, create_approximate, create_relaxed_approximate, model_attrs


def read_prep_instance(instance_name, sr = False, al = 0.25, instancepath="PrepInstances"):
    """
    A function to parse a prepared instance, which is a directory called <instance_name> that contains
    <instance_name>.lp (the overall optimization problem)
    master.lp (the master problem)
    dualsub.lp (the sub problem)
    H.json (a matrix containing the coefficients of master variables constraints that contained master and sub variables)
    order.json (a list that determines the order of the master variables)
    master_0.25.sol (a 1.25-optimal solution of the overall problem)
    optimal.sol (an optimal solution of the overall problem, not used)
    core.json (a core point for the Magnanti-Wong method)
    
    returns ret_object, a transformation of the files' contents into a structure that can be handled more conveniently
    """
    
    logger.message("Read Prep Instance")
    # initialize ret_object
    ret_object = model_attrs()

    ret_object.master_vars = master_vars = {}
    ret_object.sub_vars = sub_vars = {}

    ret_object.master_cons = master_cons = {}
    ret_object.mix_cons = mix_cons = {}
    ret_object.sub_cons = sub_cons = {}
    ret_object.all_sub_cons = all_sub_cons = {}
    ret_object.H = H = {} # one entry for each mix_constr, each entry is a |master_var|-length numpy array
    ret_object.f = f = []
    
    instance_path = f"{instancepath}/{instance_name}/"
    ret_object.m = m = grb.read(instance_path + instance_name + ".lp")
    
    master = grb.read(instance_path + "master.lp")
    master._LHS = []
    master._RHS = []
    
    # check if the LP-relaxation of the original problem is supposed to be solved
    ap_ad = ""
    if sr:
        ap_ad = "_r"
    
    # generate an approximative solution if it does not already exist
    if os.path.exists(f"{instance_path}/master_{al}{ap_ad}.sol"):
        pass
    else:
        if not sr:
            create_approximate(instance_path, instance_name, al)
        else:
            create_relaxed_approximate(instance_path, instance_name, al)
    
    # set some model parameters for the master
    master.Params.LazyConstraints = 1
    master.Params.DualReductions = 0
    master.Params.Seed = 1
    master.Params.Threads = 1
    master.Params.PreSolve = 0
    master.Params.TimeLimit = 3420 # leave the HPC three minutes for bookkeeping...
    m.read(f"{instance_path}/master_{al}{ap_ad}.sol")
    for variable in m.getVars():
        variable.lb = variable.ub = variable.start
    m.Params.OutputFlag = 1.
    # force the master problem variables to the approximative solution at the start
    m.optimize()
    
    # set the initial parameters for incumbent to the value that comes out
    master._incumbent = m.ObjVal
    
    # initialize some other parameters
    master._gap_threshold = False
    master._begin_time = time.time()
    master._vars = {}
    set_obs_attrs(master)
    
    # initialize the dual subproblem
    dualsub = grb.read(instance_path + "dualsub.lp")
    dualsub.Params.InfUnbdInfo = 1
    dualsub.Params.OutputFlag = 0
    dualsub.ModelSense = grb.GRB.MAXIMIZE
    dualsub._vars = {}
    dualsub._constrs = {}
    
    # create a dict in which all master variables are in order (master._vars)
    # create a vector that contains the master variables objective coefficients (in correct order)
    tmp_dict = {}
    for master_var in master.getVars():
        if master_var.varname != "aux" and master_var.varname != "dummy":
            tmp_dict[master_var.varname] = master_var
        else:
            master._aux = master_var
    with open(instance_path + "order.json", "r") as my_file:
        order_tmp = json.load(my_file)
    for key in order_tmp:
        master_vars[key] = tmp_dict[key]
        master._vars[key] = tmp_dict[key]
        f.append(tmp_dict[key].obj)
    del tmp_dict
    
    # set the master problem objective, and initialize the _ival attributes (the incumbent solution)
    m_obj = 0.
    for variable in m.getVars():
        try:
            master._vars[variable.varname]._ival = variable.X
            m_obj += variable.X * variable.obj
        except: pass
    master._aux._ival = master._incumbent - m_obj
    
    # create lists for master constraints, mixed constraints, and pure subproblem constraints
    for master_constr in master.getConstrs():
        master_cons[master_constr.constrname] = master_constr
    for sub_var in dualsub.getVars():
        all_sub_cons[sub_var.varname] = sub_var
        dualsub._vars[sub_var.varname] = sub_var
        sub_var._RHS = sub_var.obj
    for sub_constr in dualsub.getConstrs():
        sub_vars[sub_constr.constrname] = sub_constr
        dualsub._constrs[sub_constr.constrname] = sub_constr
    dualsub.update()
    
    # read H and transform it into a numpy array
    with open(instance_path + "H.json", "r") as my_file:
        H_tmp = json.load(my_file)
    for tmp_key, tmp_value in H_tmp.items():
        H[tmp_key] = np.array(tmp_value)
    
    # make f to an attribute of the ret_object
    ret_object.f = np.array(f)
    
    # distinguish between mixed and pure subproblem constraints
    for key in all_sub_cons:
        if key in H:
            mix_cons[key] = all_sub_cons[key]
        else:
            sub_cons[key] = all_sub_cons[key]
    
    # make all the master variables continuous if the relaxation is supposed to be solved
    if sr:
        for variable in master.getVars():
            variable.vtype = "C"
        master._dummy = master.addVar(vtype = "B", obj = 0., name = "dummy")
        master.update()
    
    # return master, dualsub and ret_object
    return master, dualsub, ret_object


def consolidate_expr(expr, factor=1e-10):
    """
    A helper function that makes gurobi LinExpr()s more compact by merging terms with the same variable
    """
    my_dict = {}
    for index in range(expr.size()):
        my_dict[expr.getVar(index)] = 0.
    for index in range(expr.size()):
        my_dict[expr.getVar(index)] += expr.getCoeff(index)
    max_coeff = np.max(list(my_dict.values()))
    ret_expr = expr.getConstant()
    for key, value in my_dict.items():
        if abs(value) >= factor * abs(max_coeff):
            ret_expr += value * key
    return ret_expr


def determine_core(master, instance_name, instancepath="PrepInstances"):
    """
    this is a rudimentary method to determine a core value of a master problem
    currently only works if NO CONSTRAINTS are given OR if core.json exists in prepared instance folder...
    """
    
    # read core from core.json if this file exists
    for _ in master.getConstrs(): # IF Constraints exist, load the core file
        with open(f"{instancepath}/{instance_name}/core.json", "r") as my_file:
            core_dict = json.load(my_file)
        for variable in master.getVars():
            try: variable._core_value = core_dict[variable.varname]
            except: print(variable.varname)
        master.update()
        return
    
    # otherwise, try to generate a core point from scratch, depending on lower and upper bounds of variables
    for var in master.getVars(): # if not, determine the core arbitrarily
        if var.ub - var.lb == 0.:
            var._core_value = var.lb
        elif var.ub - var.lb <= 1. or var.vtype == "B":
            var._core_value = var.lb + 0.5
        elif var.ub - var.lb <= 2.:
            var._core_value = var.lb + 1.
        elif var.ub - var.lb > 2:
            var._core_value = var.lb + 2.
        else:
            logger.warning("Invalid master bounds. Terminating.")
            sys.exit()
    master.update()


def modify_ds_for_mis(dualsub, ret_object, normalization = "standard"):
    """
    Modifies the dualsub model for being a subproblem compatible with the MIS strategy.
    It
    1. Adds a variable x_abs that attains |x| for variables x that are not sign-restricted
    2. Adds pi_0 to the model
    """
    logger.message("Modify Dualsub for MIS")
    
    # renaming
    master_vars = ret_object.master_vars
    sub_vars = ret_object.sub_vars
    master_cons = ret_object.master_cons
    mix_cons = ret_object.mix_cons
    sub_cons = ret_object.sub_cons
    all_sub_cons = ret_object.all_sub_cons
    H = ret_object.H
    f = ret_object.f
    m = ret_object.m
    
    # sets the lb and the ub of the "zero" variable to 0/inf. (In the classical dual sub, the "zero" variable exists, but is fixed to 1.)
    dualsub._vars["zero"].lb = 0.
    dualsub._vars["zero"].ub = dualsub._vars["zero"]._ub = float("inf")
    
    # introduce x^+, x^- if necessary, and the corresponding constraints
    for mix_constr_name, mix_constr in mix_cons.items():
        # if we have sign restricted variabes, we do nothing but renaming
        if mix_constr.ub == 0.:
            dualsub._vars[mix_constr_name + "_auxiliary"] = dualsub._vars[mix_constr_name]
            dualsub._vars[mix_constr_name]._coeff = -1.
        elif mix_constr.lb == 0.:
            dualsub._vars[mix_constr_name + "_auxiliary"] = dualsub._vars[mix_constr_name]
            dualsub._vars[mix_constr_name]._coeff = 1.
        # if not, we have to introduce x_abs and the constraints that assure x_abs = |x|
        else:
            dualsub._vars[mix_constr_name + "_auxiliary"] = dualsub.addVar(lb = 0., ub = float("inf"), obj = 0., name=f"{mix_constr_name}_auxiliary")
            dualsub._vars[mix_constr_name]._coeff = 1.
            dualsub.update()
            dualsub._constrs[f"aux_pos_{mix_constr_name}"] = dualsub.addConstr(dualsub._vars[mix_constr_name] <= dualsub._vars[mix_constr_name + "_auxiliary"])
            dualsub._constrs[f"aux_neg_{mix_constr_name}"] = dualsub.addConstr( - dualsub._vars[mix_constr_name] <= dualsub._vars[mix_constr_name + "_auxiliary"])
            dualsub.update()
    
    # different normalization variants. For the computational tests, Fischettis 'standard' has been used. Only dual Variables that correspond to 'mix' constraints.
    if normalization == "standard":
        dualsub._constrs["zero"] = dualsub.addConstr( sum(dualsub._vars[mix_constr_name]._coeff * dualsub._vars[mix_constr_name + "_auxiliary"] for mix_constr_name in mix_cons) + dualsub._vars["zero"] == 1., name="zero")
    elif normalization == "normalized":
        dualsub._constrs["zero"] = dualsub.addConstr( sum((dualsub._vars[mix_constr_name]._coeff * dualsub._vars[mix_constr_name + "_auxiliary"]) * np.linalg.norm(H[mix_constr_name]) for mix_constr_name in mix_cons) + 1000. * dualsub._vars["zero"] == 1000., name="zero")
    elif normalization == "RHS":
        dualsub._constrs["zero"] = dualsub.addConstr( sum((dualsub._vars[mix_constr_name]._coeff * dualsub._vars[mix_constr_name + "_auxiliary"]) * (abs(mix_constr._RHS) or 1.) for mix_constr_name in mix_cons) + dualsub._vars["zero"] == 1., name="zero")
    elif normalization == "perturbed":
        dualsub._constrs["zero"] = dualsub.addConstr( sum(dualsub._vars[mix_constr_name]._coeff * dualsub._vars[mix_constr_name + "_auxiliary"] * 10**(- np.random.randint(7)) for mix_constr_name in mix_cons) + dualsub._vars["zero"] == 1., name="zero")
    elif normalization == "perturbed2":
        dualsub._constrs["zero"] = dualsub.addConstr( sum(dualsub._vars[mix_constr_name]._coeff * dualsub._vars[mix_constr_name + "_auxiliary"] * 10**(- 5 * np.random.randint(2)) for mix_constr_name in mix_cons) + dualsub._vars["zero"] == 1., name="zero")
    elif normalization == "perturbed3":
        dualsub._constrs["zero"] = dualsub.addConstr( sum(dualsub._vars[mix_constr_name]._coeff * dualsub._vars[mix_constr_name + "_auxiliary"] * 10**(- 5 * int(np.random.random(1) >= 0.95)) for mix_constr_name in mix_cons) + dualsub._vars["zero"] == 1., name="zero")
    dualsub.update()


def modify_ds_mis_to_ols(dualsub, ret_object):
    # this is needed for the hybrid approach
    logger.message("Modify MIS to OLS")
    master_vars = ret_object.master_vars
    sub_vars = ret_object.sub_vars
    master_cons = ret_object.master_cons
    mix_cons = ret_object.mix_cons
    sub_cons = ret_object.sub_cons
    all_sub_cons = ret_object.all_sub_cons
    H = ret_object.H
    f = ret_object.f
    m = ret_object.m
    dualsub.remove(dualsub._constrs["zero"])
    for mix_constr_name, mix_constr in mix_cons.items():
        # changed mix-constr.lb to mix_constr.lb. Why has this ever worked!!???
        if mix_constr.ub != 0. and mix_constr.lb != 0.:
            dualsub.remove(dualsub._constrs[f"aux_pos_{mix_constr_name}"])
            dualsub.remove(dualsub._constrs[f"aux_neg_{mix_constr_name}"])
    dualsub.update()
    for mix_constr_name, mix_constr in mix_cons.items():
        if mix_constr.ub != 0. and mix_constr.lb != 0.:
            dualsub.remove(dualsub._vars[mix_constr_name + "_auxiliary"])
    dualsub.update()
    dualsub._constrs["zero"] = dualsub.addConstr(dualsub._vars["zero"] == 1., name="zero")
    dualsub.update()
            

def modify_ds_for_facet(dualsub, ret_object):
    """
    Modifies the dualsub model for being a subproblem compatible with the facet strategy.
    It
    1. Modifies the bounds for pi_0 ("zero") and adds a normalization constraint ("zero", as well) to the sub.
    This constraint is supposed to be adapted during the Benders callbacks.
    """
    logger.message("Modify Dualsub for FACET")
    
    # adapt bounds
    dualsub._vars["zero"].lb = 0.
    dualsub._vars["zero"].ub = dualsub._vars["zero"]._ub = float("inf")
    # add constraint
    dualsub._constrs["zero"] = dualsub.addConstr( dualsub._vars["zero"] == 1., name="zero")
    dualsub.update()


def modify_ds_for_ols(dualsub, ret_object):
    """
    Modifies the dualsub model for being a subproblem compatible with the OLS strategy.
    It
    1. Modifies the bounds for pi_0 ("zero") and adds a normalization constraint ("zero", as well) to the sub.
    This constraint is supposed to be adapted during the Benders callbacks.
    NOTE: it does exactly the same as modify_ds_for_facet; one of the two could be removed.
    """
    logger.message("Modify Dualsub for OLS")
    
    # adapt bounds
    dualsub._vars["zero"].lb = 0.
    dualsub._vars["zero"].ub = dualsub._vars["zero"]._ub = float("inf")
    # add constraint
    dualsub._constrs["zero"] = dualsub.addConstr( dualsub._vars["zero"] == 1., name="zero")
    dualsub.update()


def adapt_dual_sub(master, dualsub, ret_object, value_method = lambda var: var.X):
    """
    Gets an iterate (the variables of the input object "master") and adapts the objective function coefficients of the
    dual subproblem accordingly.
    """
    mix_cons = ret_object.mix_cons
    H = ret_object.H
    x_vec = np.array([value_method(var) for varname, var in master._vars.items()])
    for mix_constr_name, mix_constr in mix_cons.items():
        dualsub._vars[mix_constr_name].obj = mix_constr._RHS - H[mix_constr_name].T @ x_vec
    dualsub.update()


def adapt_mis_dual_sub(master, dualsub, ret_object, value_method = lambda var: var.X):
    """
    Gets an iterate (the variables of the input object "master") and adapts the objective function coefficients of the
    dual subproblem accordingly. Also adapts the objective function coefficient of pi_0.
    """
    adapt_dual_sub(master, dualsub, ret_object, value_method = value_method)
    dualsub._vars["zero"].obj = - value_method(master._aux)
    dualsub.update()


def facet_normalization(master, dualsub, ret_object, value_method = lambda var: var.X):
    """
    Adapts the facet normalization constraint to the dual subproblem; also sets the objective function coefficient for pi_0.
    """
    mix_cons = ret_object.mix_cons
    H = ret_object.H
    f = ret_object.f
    # x_bar and x_tilde
    x_bar = np.array([value_method(var) for varname, var in master._vars.items()])
    x_diff = np.array([var._ival for varname, var in master._vars.items()]) - x_bar
    aux_diff = master._aux._ival + 1. - value_method(master._aux)
    # aux variable objective
    dualsub._vars["zero"].obj = - value_method(master._aux)
    # setup new normalization constraint
    dualsub.remove(dualsub._constrs["zero"])
    stop2 = time.time()
    dualsub._constrs["zero"] = dualsub.addConstr( sum(dualsub._vars[mix_constr_name] * (H[mix_constr_name].T @ x_diff) for mix_constr_name in mix_cons) + dualsub._vars["zero"] * aux_diff == 1., name = "zero")
    master._ntime += time.time() - stop2
    dualsub.update()


def ols_pre_normalization(master, dualsub, ret_object, value_method = lambda var: var.X):
    """
    Modifies the bounds of pi_0 and the normalization constraint in a way that the subproblem gets a standard Benders subproblem.
    """
    mix_cons = ret_object.mix_cons
    H = ret_object.H
    f = ret_object.f
    dualsub._vars["zero"].obj = 0.
    dualsub._vars["zero"].lb = 1.
    dualsub._vars["zero"].ub = 1.
    # remove normalization constraint
    dualsub.remove(dualsub._constrs["zero"])
    dualsub._constrs["zero"] = dualsub.addConstr(dualsub._vars["zero"] == 1, name = "zero")
    dualsub.update()


def ols_standard_normalization(master, dualsub, ret_object, value_method = lambda var: var.X):
    """
    Adapts the OLS normalization constraint to the dual subproblem; also sets the objective function coefficient for pi_0.
    """
    mix_cons = ret_object.mix_cons
    H = ret_object.H
    f = ret_object.f
    # x_bar and x_tilde
    x_bar = np.array([value_method(var) for varname, var in master._vars.items()])
    x_diff = np.array([var._ival for varname, var in master._vars.items()]) - x_bar
    # aux variable objective
    dualsub._vars["zero"].obj = f.T @ x_bar - master._incumbent
    dualsub._vars["zero"].lb = 0.
    dualsub._vars["zero"].ub = dualsub._vars["zero"]._ub
    # setup new normalization constraint
    dualsub.remove(dualsub._constrs["zero"])
    stop2 = time.time()
    dualsub._constrs["zero"] = dualsub.addConstr( sum(dualsub._vars[mix_constr_name] * (H[mix_constr_name].T @ x_diff) for mix_constr_name in mix_cons) - dualsub._vars["zero"] * (f.T @ x_diff) == 1000., name = "zero")
    master._ntime += time.time() - stop2
    dualsub.update()


def cb_classic_benders(model, where, dualsub, ret_object, begin_time):
    """
    Callback, calculating a standard Benders optimality/feasibility cut;
    Called by Model.optimize() whenever a new integer solution is determined by B&C.
    """
    if where == grb.GRB.Callback.MIPSOL:
        
        # handle the input
        master_vars = ret_object.master_vars
        sub_vars = ret_object.sub_vars
        master_cons = ret_object.master_cons
        mix_cons = ret_object.mix_cons
        sub_cons = ret_object.sub_cons
        all_sub_cons = ret_object.all_sub_cons
        H = ret_object.H
        f = ret_object.f
        m = ret_object.m
        
        # increase cut counter by 1
        model._count += 1
        
        # calculate raw master solution
        master_val = 0.
        for m_var_name, m_var in master_vars.items():
            master_val += model.cbGetSolution(model._vars[m_var_name]) * master_vars[m_var_name].obj
        
        # adapt dual subproblem with iterate and solve
        adapt_dual_sub(model, dualsub, ret_object, value_method = model.cbGetSolution)
        dualsub.optimize()
        
        # if it is not unbounded, insert an optimality cut, and update the incubment if necessary
        if dualsub.status == 2:
            # add cut
            x_vec = np.array([var for var_name, var in model._vars.items()])
            model.cbLazy(model._aux >= consolidate_expr(sum(dualsub._vars[sub_constr_name].X * sub_constr._RHS for sub_constr_name, sub_constr in sub_cons.items()) +
                                        sum(dualsub._vars[mix_constr_name].X * (mix_constr._RHS - H[mix_constr_name].T @ x_vec ) for mix_constr_name, mix_constr in mix_cons.items())))
            # save new incumbent
            master_val += dualsub.objval
            if master_val < model._incumbent:
                model._incumbent = master_val
            logger.message(f"Ins. O-Cut. SubValue {dualsub.objval:4.1f}. OverallValue {master_val:4.1f}. Incumbent {model._incumbent:4.1f}.")
        
        # otherwise, insert a feasibility cut
        elif dualsub.status == 5:
            # add cut
            x_vec = np.array([var for var_name, var in model._vars.items()])
            x_bar = np.array([model.cbGetSolution(var) for var_name, var in model._vars.items()])
            model.cbLazy(0. >= consolidate_expr(sum(dualsub._vars[sub_constr_name].UnbdRay * sub_constr._RHS for sub_constr_name, sub_constr in sub_cons.items()) + sum(dualsub._vars[mix_constr_name].UnbdRay * (mix_constr._RHS - H[mix_constr_name].T @ x_vec ) for mix_constr_name, mix_constr in mix_cons.items())))
            logger.message(f"Ins. F-Cut. SubValue ------. OverallValue ------. Incumbent {model._incumbent}.")
        
        # catch cases that should not happen
        else:
            logger.warning(f"Classic subproblem has invalid status {dualsub.status}.")
            input()
        
        # terminate if over time limit
        if time.time() - begin_time >= 3420.:
            model.terminate()


def cb_cpx_ols_benders(model, where, dualsub, ret_object, begin_time):
    """
    Callback, calculating a standard Benders optimality cut or a CW-normalized feasibility cut;
    Called by Model.optimize() whenever a new integer solution is determined by B&C.
    """
    if where == grb.GRB.Callback.MIPSOL:
        
        # handle the input
        master_vars = ret_object.master_vars
        sub_vars = ret_object.sub_vars
        master_cons = ret_object.master_cons
        mix_cons = ret_object.mix_cons
        sub_cons = ret_object.sub_cons
        all_sub_cons = ret_object.all_sub_cons
        H = ret_object.H
        m = ret_object.m
        f = ret_object.f
        
        # increase cut counter by 1
        model._count += 1
        
        # store iterate, core and variable vector: x_bar, x_tilde, x_var (for cuts)
        x_bar = np.array([model.cbGetSolution(var) for varname, var in model._vars.items()])
        x_tilde = np.array([var._ival for varname, var in model._vars.items()])
        x_var = np.array([var for varname, var in model._vars.items()])
        
        # adapt dual subproblem with iterate
        stop = time.time()
        adapt_dual_sub(model, dualsub, ret_object, value_method = model.cbGetSolution)
        model._atime += time.time() - stop
        stop = time.time()
        
        # adapt the subproblem that it generates a standard Benders optimality cut, and solve
        oc_begin = time.time()
        ols_pre_normalization(model, dualsub, ret_object, value_method = model.cbGetSolution)
        dualsub.optimize()
        model._oc_time += time.time() - oc_begin
        
        # we check if the standard dualsub is unbounded;
        valid = (dualsub.status == 2)
        
        # if not:
        if valid:
            cut_type = "Opti"
        
        # if it is unbounded:
        else:
            cut_type = "Feas"
        
        # if valid, we are in the case where we just insert the benders optimality cut
        if valid:
            # update the incumbent and its value if necessary
            if dualsub.objval + f.T @ x_bar <= model._incumbent + 1e-04:
                model._incumbent = f.T @ x_bar + dualsub.objval
                for var in model.getVars():
                    var._ival = model.cbGetSolution(var)
                model._aux._ival = model.cbGetSolution(model._aux)
                logger.message(f"Inserting Improvement-Cut. New incumbent {model._incumbent:4.1f}.")
            
            # insert the cut
            model.cbLazy(dualsub._vars["zero"].X * model._aux >= consolidate_expr(sum(dualsub._vars[sub_constr_name].X * sub_constr._RHS for sub_constr_name, sub_constr in sub_cons.items()) + sum(dualsub._vars[mix_constr_name].X * (mix_constr._RHS - H[mix_constr_name].T @ x_var ) for mix_constr_name, mix_constr in mix_cons.items())))
            logger.message(f"Ins. OPT-Cut. Incumbent: {model._incumbent:4.1f}.")
        
        # in this case the standard dual subproblem is infeasible; we calculate and insert an OLS cut
        else:
            stop = time.time()
            # insert the OLS normalization constraint into the dual subproblem and optimize
            ols_standard_normalization(model, dualsub, ret_object, value_method = model.cbGetSolution)
            dualsub.optimize()
            
            if dualsub.status == 2:
                # insert cut
                model.cbLazy(dualsub._vars["zero"].X * model._aux >= consolidate_expr(sum(dualsub._vars[sub_constr_name].X * sub_constr._RHS for sub_constr_name, sub_constr in sub_cons.items()) + sum(dualsub._vars[mix_constr_name].X * (mix_constr._RHS - H[mix_constr_name].T @ x_var ) for mix_constr_name, mix_constr in mix_cons.items())))
                model._icount += 1
            
            # catch cases that should not happen
            else:
                logger.warning(f"CW subproblem negative/infeasible and Standard subproblem infeasible.")
                input()
            logger.message(f"Ins. OLS-Cut instead of FEAS-Cut. Incumbent: {model._incumbent:4.1f}.")
            model._itime += time.time() - stop
        
        # terminate if over time limit
        if time.time() - begin_time >= 3420.:
            model.terminate()


def cb_cpx_benders(model, where, dualsub, ret_object, begin_time):
    """
    Callback, calculating a standard Benders optimality cut or an OLS-normalized cut
    to cut off iterates with infeasible subproblem;
    Called by Model.optimize() whenever a new integer solution is determined by B&C.
    """
    if where == grb.GRB.Callback.MIPSOL:
        
        # handle the input
        master_vars = ret_object.master_vars
        sub_vars = ret_object.sub_vars
        master_cons = ret_object.master_cons
        mix_cons = ret_object.mix_cons
        sub_cons = ret_object.sub_cons
        all_sub_cons = ret_object.all_sub_cons
        H = ret_object.H
        m = ret_object.m
        f = ret_object.f
        
        # increase cut counter by 1
        model._count += 1
        
        # store iterate, core and variable vector: x_bar, x_tilde, x_var (for cuts)
        x_bar = np.array([model.cbGetSolution(var) for varname, var in model._vars.items()])
        x_tilde = np.array([var._ival for varname, var in model._vars.items()])
        x_var = np.array([var for varname, var in model._vars.items()])
        
        # adapt dual subproblem with iterate
        stop = time.time()
        adapt_dual_sub(model, dualsub, ret_object, value_method = model.cbGetSolution)
        model._atime += time.time() - stop
        stop = time.time()
        
        # adapt the subproblem that it generates a standard Benders optimality cut, and solve
        oc_begin = time.time()
        ols_pre_normalization(model, dualsub, ret_object, value_method = model.cbGetSolution)
        dualsub.optimize()
        model._oc_time += time.time() - oc_begin
        
        # we check if the standard dualsub is unbounded;
        valid = (dualsub.status == 2)
        
        # if not:
        if valid:
            cut_type = "Opti"
        
        # if it is unbounded:
        else:
            cut_type = "Feas"
        
        # if valid, we are in the case where we just insert the benders optimality cut
        if valid:
            # we replace _ival of variabes if the current iterate
            # improves the incumbent...
            if dualsub.objval + f.T @ x_bar <= model._incumbent + 1e-04:
                model._incumbent = f.T @ x_bar + dualsub.objval
                for var in model.getVars():
                    var._ival = model.cbGetSolution(var)
                model._aux._ival = model.cbGetSolution(model._aux)
                logger.message(f"Inserting Improvement-Cut. New incumbent {model._incumbent:4.1f}.")
            
            # insert the cut
            model.cbLazy(dualsub._vars["zero"].X * model._aux >= consolidate_expr(sum(dualsub._vars[sub_constr_name].X * sub_constr._RHS for sub_constr_name, sub_constr in sub_cons.items()) + sum(dualsub._vars[mix_constr_name].X * (mix_constr._RHS - H[mix_constr_name].T @ x_var ) for mix_constr_name, mix_constr in mix_cons.items())))
            logger.message(f"Ins. OPT-Cut. Incumbent: {model._incumbent:4.1f}.")
        
        # in this case the standard dual subproblem is infeasible; we calculate and insert an OLS cut
        else:
            stop = time.time()
            ols_standard_normalization(model, dualsub, ret_object, value_method = model.cbGetSolution)
            # CW normaliziation is OLS normalization with pi_0/alpha restricted to 0
            # for no reason we restrict ourselves to feasibility cuts here.
            dualsub._vars["zero"].ub = 0.
            dualsub.optimize()
            if dualsub.status == 2:
                # insert cut
                model.cbLazy(dualsub._vars["zero"].X * model._aux >= consolidate_expr(sum(dualsub._vars[sub_constr_name].X * sub_constr._RHS for sub_constr_name, sub_constr in sub_cons.items()) + sum(dualsub._vars[mix_constr_name].X * (mix_constr._RHS - H[mix_constr_name].T @ x_var ) for mix_constr_name, mix_constr in mix_cons.items())))
                model._icount += 1
            
            # catch cases that should not happen
            else:
                logger.warning(f"OLS subproblem negative/infeasible and Standard subproblem infeasible.")
                input()
            logger.message(f"Ins. OLS-Cut instead of FEAS-Cut. Incumbent: {model._incumbent:4.1f}.")
            model._itime += time.time() - stop
        
        # terminate if over time limit
        if time.time() - begin_time >= 3420.:
            model.terminate()


def cb_mw_benders(model, where, dualsub, ret_object, begin_time):
    """
    Callback, calculating a Magnanti-Wong cut;
    Called by Model.optimize() whenever a new integer solution is determined by B&C.
    """
    if where == grb.GRB.Callback.MIPSOL:
        
        # handle the input
        master_vars = ret_object.master_vars
        sub_vars = ret_object.sub_vars
        master_cons = ret_object.master_cons
        mix_cons = ret_object.mix_cons
        sub_cons = ret_object.sub_cons
        all_sub_cons = ret_object.all_sub_cons
        H = ret_object.H
        f = ret_object.f
        m = ret_object.m
        
        # increase cut counter by 1
        model._count += 1
        
        # calculate raw master solution
        master_val = 0.
        for m_var_name, m_var in master_vars.items():
            master_val += model.cbGetSolution(model._vars[m_var_name]) * master_vars[m_var_name].obj
        
        # adapt dual subproblem with iterate and solve
        adapt_dual_sub(model, dualsub, ret_object, value_method = model.cbGetSolution)
        dualsub.optimize()
        
        # if it is not unbounded, insert an optimality cut, and update the incubment if necessary
        if dualsub.status == 2:
            
            # but first optimize at the core point:
            # insert temporary constraint and change objective function and re-optimize
            dualsub._objval = dualsub.objval
            dualsub._tmp = dualsub.addConstr(sum(var * var.obj for var in dualsub.getVars()) >= dualsub.objval)
            adapt_dual_sub(model, dualsub, ret_object, value_method = lambda var: var._core_value)
            dualsub.update()
            dualsub.optimize()
            
            # optimality cut
            x_vec = np.array([var for var_name, var in model._vars.items()])
            model.cbLazy(model._aux >= consolidate_expr(sum(dualsub._vars[sub_constr_name].X * sub_constr._RHS for sub_constr_name, sub_constr in sub_cons.items()) + sum(dualsub._vars[mix_constr_name].X * (mix_constr._RHS - H[mix_constr_name].T @ x_vec ) for mix_constr_name, mix_constr in mix_cons.items())))
            
            # update incumbent if necessary
            if master_val + dualsub._objval <= model._incumbent:
                model._incumbent = master_val + dualsub._objval
            
            logger.message(f"Ins. O-Cut. SubValue core {dualsub.objval:4.1f} point {dualsub._objval:4.1f}. OverallValue {master_val:4.1f}. Incumbent {model._incumbent:4.1f}.")
            dualsub.remove(dualsub._tmp)
            dualsub.update()
        
        # otherwise, insert a feasibility cut
        elif dualsub.status == 5:
            
            # insert cut
            x_vec = np.array([var for var_name, var in model._vars.items()])
            model.cbLazy(0. >= consolidate_expr(sum(dualsub._vars[sub_constr_name].UnbdRay * sub_constr._RHS for sub_constr_name, sub_constr in sub_cons.items()) + sum(dualsub._vars[mix_constr_name].UnbdRay * (mix_constr._RHS - H[mix_constr_name].T @ x_vec ) for mix_constr_name, mix_constr in mix_cons.items())))
            logger.message(f"Ins. F-Cut. SubValue ------. OverallValue ------. Incumbent {model._incumbent:4.1f}.")
        
        # catch cases that should not happen
        else:
            logger.warning(f"MW subproblem has invalid status {dualsub.status}.")
            input()
        
        # terminate if over time limit
        if time.time() - begin_time >= 3420.:
            model.terminate()


def cb_mis_benders(model, where, dualsub, ret_object, begin_time, hybrid = False):
    """
    Callback, calculating a MIS cut;
    Called by Model.optimize() whenever a new integer solution is determined by B&C.
    """
    if where == grb.GRB.Callback.MIPSOL:
        
        # handle the input
        master_vars = ret_object.master_vars
        sub_vars = ret_object.sub_vars
        master_cons = ret_object.master_cons
        mix_cons = ret_object.mix_cons
        sub_cons = ret_object.sub_cons
        all_sub_cons = ret_object.all_sub_cons
        H = ret_object.H
        f = ret_object.f
        m = ret_object.m
        
        # increase cut counter by 1
        model._count += 1
        
        # calculate raw master solution
        master_val = 0.
        for m_var_name, m_var in master_vars.items():
            master_val += model.cbGetSolution(model._vars[m_var_name]) * master_vars[m_var_name].obj
        
        # adapt dual subproblem with iterate and solve
        adapt_mis_dual_sub(model, dualsub, ret_object, value_method = model.cbGetSolution)
        dualsub.optimize()
        
        
        if dualsub.status == 2:
            # MIS cut
            if dualsub.objval <= 1e-10:
                logger.message(f"New MIS Inc. SubValue {dualsub.objval:4.1f}. Incumbent {model.cbGet(grb.GRB.Callback.MIPSOL_OBJBST)}.")
                objbst = model._incumbent = min(model._incumbent, model.cbGet(grb.GRB.Callback.MIPSOL_OBJBST))
                for var in model.getVars():
                    var._ival = model.cbGetSolution(var)
                model._aux._ival = model.cbGetSolution(model._aux)
                objbnd = model.cbGet(grb.GRB.Callback.MIPSOL_OBJBND)
                # switch to OLS cuts if the callback is used by the hybrid strategy
                model._gap_threshold = ((abs(objbst - objbnd) / (1.0 + abs(objbst))) < 0.1) and hybrid and (model._count >= 100)
                
            x_vec = np.array([var for var_name, var in model._vars.items()])
            model.cbLazy(dualsub._vars["zero"].X * model._aux >= consolidate_expr(sum(dualsub._vars[sub_constr_name].X * sub_constr._RHS for sub_constr_name, sub_constr in sub_cons.items()) + sum(dualsub._vars[mix_constr_name].X * (mix_constr._RHS - H[mix_constr_name].T @ x_vec ) for mix_constr_name, mix_constr in mix_cons.items())))
            logger.message(f"Ins. MIS-Cut. SubValue {dualsub.objval:4.1f}. MasterValue {master_val:4.1f}.")
            if model._gap_threshold:
                print("Switching to OLS")
                modify_ds_mis_to_ols(dualsub, ret_object)
        
        # catch cases that should not happen
        else:
            logger.warning(f"MIS subproblem has invalid status {dualsub.status}.")
            input()
        
        # terminate if over time limit
        if time.time() - begin_time >= 3420.:
            model.terminate()


def cb_facet_benders(model, where, dualsub, ret_object, begin_time):
    """
    Callback, calculating a Facet cut;
    Called by Model.optimize() whenever a new integer solution is determined by B&C.
    """
    if where == grb.GRB.Callback.MIPSOL:
        
        # handle the input
        master_vars = ret_object.master_vars
        sub_vars = ret_object.sub_vars
        master_cons = ret_object.master_cons
        mix_cons = ret_object.mix_cons
        sub_cons = ret_object.sub_cons
        all_sub_cons = ret_object.all_sub_cons
        H = ret_object.H
        m = ret_object.m
        f = ret_object.f
        
        # increase cut counter by 1
        model._count += 1
        
        # store iterate, core and variable vector: x_bar, x_tilde, x_var (for cuts)
        x_var = np.array([var for varname, var in model._vars.items()])
        x_tilde = np.array([var._ival for varname, var in model._vars.items()])
        x_bar = np.array([model.cbGetSolution(var) for varname, var in model._vars.items()])
        # as well as the corr. values for eta...
        aux_tilde = model._aux._ival + 1.
        aux_bar = model.cbGetSolution(model._aux)
        
        # adapt the sub to have the correct objective function...
        stop = time.time()
        adapt_dual_sub(model, dualsub, ret_object, value_method = model.cbGetSolution)
        model._atime += time.time() - stop
        stop = time.time()
        
        # adapt the facet normalization constraint and optimize
        facet_normalization(model, dualsub, ret_object, value_method = model.cbGetSolution)
        dualsub.optimize()
        if dualsub.status == 2:
            # add the cut
            model.cbLazy(dualsub._vars["zero"].X * model._aux >= consolidate_expr(sum(dualsub._vars[sub_constr_name].X * sub_constr._RHS for sub_constr_name, sub_constr in sub_cons.items()) + sum(dualsub._vars[mix_constr_name].X * (mix_constr._RHS - H[mix_constr_name].T @ x_var ) for mix_constr_name, mix_constr in mix_cons.items())))
            logger.message(f"Ins. FACET-Cut.")
            
            # update the core point
            x_new = x_new = x_bar + dualsub.objval * (x_tilde - x_bar)
            
            # store the core point as model variable attributes
            for var, value in zip(x_var, x_new):
                var._ival = value
            model._aux._ival = aux_bar + dualsub.objval * (aux_tilde - aux_bar)
        
        # catch cases that should not happen
        else:
            logger.warning(f"Facet subproblem has invalid status {dualsub.status}.")
            input()
        
        # terminate if over time limit
        if time.time() - begin_time >= 3420.:
            model.terminate()


def cb_ols_benders(model, where, dualsub, ret_object, begin_time):
    """
    Callback, calculating an OLS cut; if this fails, a standard Benders cut is calculated.
    Called by Model.optimize() whenever a new integer solution is determined by B&C.
    """
    if where == grb.GRB.Callback.MIPSOL:
        
        # handle the input
        master_vars = ret_object.master_vars
        sub_vars = ret_object.sub_vars
        master_cons = ret_object.master_cons
        mix_cons = ret_object.mix_cons
        sub_cons = ret_object.sub_cons
        all_sub_cons = ret_object.all_sub_cons
        H = ret_object.H
        m = ret_object.m
        f = ret_object.f
        
        # increase cut counter by 1
        model._count += 1
        
        # store iterate, core and variable vector: x_bar, x_tilde, x_var (for cuts)
        x_bar = np.array([model.cbGetSolution(var) for varname, var in model._vars.items()])
        x_tilde = np.array([var._ival for varname, var in model._vars.items()])
        x_var = np.array([var for varname, var in model._vars.items()])
        
        # adapt the sub to have the correct objective function...
        stop = time.time()
        adapt_dual_sub(model, dualsub, ret_object, value_method = model.cbGetSolution)
        model._atime += time.time() - stop
        stop = time.time()
        ols_standard_normalization(model, dualsub, ret_object, value_method = model.cbGetSolution)
        dualsub.optimize()
        
        # check the conditions acc. Lemma 3.17: If the dualsub is infeasible,
        # or if its value is not positive (+ numerical tolerances)
        # we go over to a standard Benders cut...
        valid = (dualsub.status == 2)
        if valid:
            valid = (dualsub.objval >= 1e-04)
        
        # if all this is not the case, i.e., the obj value of the dualsub is > 1e-04, we generate
        # an OLS cut
        if valid:
            # some statistics
            model._mdcount += 1
            if abs(dualsub._vars["zero"].X - dualsub._vars["zero"].ub) <= 1e-04:
                model._alpha_at_max += 1
            if abs(dualsub.objval - 1000.) <= 1e-03:
                model._depth_at_max += 1
            if model._maxalpha < dualsub._vars["zero"].X:
                model._maxalpha = dualsub._vars["zero"].X
            if dualsub._vars["zero"].X != 0.:
                cut_type = "Opti"
            else:
                cut_type = "Feas"
            
            # add cut
            model.cbLazy(dualsub._vars["zero"].X * model._aux >= consolidate_expr(sum(dualsub._vars[sub_constr_name].X * sub_constr._RHS for sub_constr_name, sub_constr in sub_cons.items()) + sum(dualsub._vars[mix_constr_name].X * (mix_constr._RHS - H[mix_constr_name].T @ x_var ) for mix_constr_name, mix_constr in mix_cons.items())))
            logger.message(f"Ins. MD-{cut_type}-Cut. Incumbent: {model._incumbent:4.1f}. Depth: {dualsub.objval:4.1f}. Alpha: {dualsub._vars['zero'].X:4.7f}")
            
            # update the core; gamma is set to 0.5
            factor = 1. - (0.5 + 0.5 * random.random()) * (1. - dualsub.objval / 1000.)
            x_new = x_bar + factor * (x_tilde - x_bar)
            for var, value in zip(x_var, x_new):
                var._ival = value
            model._mdtime += time.time() - stop
        
        # otherwise, generate a standard Benders optimality cut
        else:
            stop = time.time()
            
            # adapt the sub to have the correct objective function and solve
            ols_pre_normalization(model, dualsub, ret_object, value_method = model.cbGetSolution)
            dualsub.optimize()
            
            # the dualsub should have an optimal solution, guaranteed by Lemma 3.17
            if dualsub.status == 2:
                # ... and it must be better than the incumbent, also guaranteed by Lemma 3.17
                if dualsub.objval + f.T @ x_bar <= model._incumbent + 1e-04: 
                    # insert cut
                    model.cbLazy(model._aux >= consolidate_expr(sum(dualsub._vars[sub_constr_name].X * sub_constr._RHS for sub_constr_name, sub_constr in sub_cons.items()) + sum(dualsub._vars[mix_constr_name].X * (mix_constr._RHS - H[mix_constr_name].T @ x_var ) for mix_constr_name, mix_constr in mix_cons.items())))
                    # update incumbent of master and ival of master variables (i.e. the core point)
                    model._incumbent = f.T @ x_bar + dualsub.objval
                    for var in model.getVars():
                        var._ival = model.cbGetSolution(var)
                    model._aux._ival = model.cbGetSolution(model._aux)
                    logger.message(f"Inserting Improvement-Cut. New incumbent {model._incumbent:4.1f}.")
                    # some statistics
                    model._icount += 1
                
                # catch cases that should not happen
                else:
                    print("Standard Benders Cut generation failed in OLS Callback.")
            
            # catch cases that should not happen
            else:
                logger.warning(f"OLS subproblem negative/infeasible and Standard subproblem infeasible.")
                input()
            model._itime += time.time() - stop
        if time.time() - begin_time >= 3420.:
            model.terminate()


def cb_hybrid_benders(model, where, dualsub, ret_object, begin_time):
    """
    Callback for the hybrid strategy. Calls MIS callback or OLS callback, depending on the model attribute _gap_threshold,
    which is changed by the MIS callback if the conditions are fulfilled (100 or more cuts generated, gap low enough)
    """
    if model._gap_threshold:
        cb_ols_benders(model, where, dualsub, ret_object, begin_time)
        
    else:
        cb_mis_benders(model, where, dualsub, ret_object, begin_time, True)


def log_csv_line(instance_name, method, master):
    """
    A method that writes out the results of an optimization run.
    """
    if master.status == 2:
        soltime = master._soltime
        oc_time = master._oc_time
        gap = 0.01
        value = master.objval
        bound = master.ObjBound
        
    elif master.status in [9, 11]: # Time limit or terminated...
        soltime = 3420.
        oc_time = master._oc_time
        try: gap = (master.objval - master.objbound) / master.objval * 100.
        except: gap = float('inf')
        try: value = master.objval
        except: value = float('inf')
        try: bound = master.objbound
        except: bound = -float('inf')
    else:
        return "{instance_name},{method},Invalid\n"
    sorting_key = f"{instance_name}_{360000 * (gap - 0.01) + soltime:.2f}"
    return f"{instance_name},{method},{soltime:4.2f},{gap:4.2f},{master._count},{value:4.2f},{bound:4.2f},{oc_time:4.2f},{sorting_key}\n"


def classical_benders(instance_name, al = 0.25, sr = False, logpath="log.csv", instancepath="PrepInstances/"):
    """
    Method to optimize a model with classical Benders cuts
    """
    # parse (prepared) instance
    master, dualsub, ret_object = read_prep_instance(instance_name, sr = sr, al = al, instancepath=instancepath)
    logger.message("Start Optimization (Classical Benders)")
    begin_time = time.time()
    # optimize the master with the callback according to the used Cut Selection Strategy
    master.optimize(lambda model, where: cb_classic_benders(model, where, dualsub, ret_object, begin_time))
    master._soltime = time.time() - begin_time
    # write output
    to_write = log_csv_line(instance_name, "clb", master)
    with open(instancepath+"/"+instance_name+"/clb.line", "w") as line_file:
        line_file.write(to_write)
    with open(logpath, "a") as my_file:
        my_file.write(to_write)


def cpx_ols_benders(instance_name, al = 0.25, sr = False, logpath="log.csv", instancepath="PrepInstances/"):
    """
    Method to optimize a model with classical Benders optimality cuts and OLS cuts if the classical dual subproblem is infeasible
    """
    # parse (prepared) instance
    master, dualsub, ret_object = read_prep_instance(instance_name, sr = sr, al = al, instancepath=instancepath)
    modify_ds_for_ols(dualsub, ret_object)
    logger.message("Start Optimization (CPX Recipe + OLS)")
    begin_time = time.time()
    # optimize the master with the callback according to the used Cut Selection Strategy
    master.optimize(lambda model, where: cb_cpx_ols_benders(model, where, dualsub, ret_object, begin_time))
    master._soltime = time.time() - begin_time
    print(f"Sol Time: {master._soltime:10.2f}s; Time spent for OC: {master._oc_time:10.2f}s; Diff {master._soltime - master._oc_time:10.2f}s")
    # write output
    to_write = log_csv_line(instance_name, "cpo", master)
    with open(instancepath+"/"+instance_name+"/cpo.line", "w") as line_file:
        line_file.write(to_write)
    with open(logpath, "a") as my_file:
        my_file.write(to_write)


def cpx_benders(instance_name, al = 0.25, sr = False, logpath="log.csv", instancepath="PrepInstances/"):
    """
    Method to optimize a model with classical Benders optimality cuts and CW normalized feasibility cuts
    """
    # parse (prepared) instance
    master, dualsub, ret_object = read_prep_instance(instance_name, sr = sr, al = al, instancepath=instancepath)
    modify_ds_for_ols(dualsub, ret_object)
    logger.message("Start Optimization (CPX Recipe)")
    begin_time = time.time()
    # optimize the master with the callback according to the used Cut Selection Strategy
    master.optimize(lambda model, where: cb_cpx_benders(model, where, dualsub, ret_object, begin_time))
    master._soltime = time.time() - begin_time
    print(f"Sol Time: {master._soltime:10.2f}s; Time spent for OC: {master._oc_time:10.2f}s; Diff {master._soltime - master._oc_time:10.2f}s")
    # write output
    to_write = log_csv_line(instance_name, "cpx", master)
    with open(instancepath+"/"+instance_name+"/cpx.line", "w") as line_file:
        line_file.write(to_write)
    with open(logpath, "a") as my_file:
        my_file.write(to_write)


def magnanti_wong_benders(instance_name, al = 0.25, sr = False, logpath="log.csv", instancepath="PrepInstances/"):
    """
    Method to optimize a model with Magnanti-Wong Benders cuts
    """
    # parse (prepared) instance
    master, dualsub, ret_object = read_prep_instance(instance_name, sr = sr, al = al, instancepath=instancepath)
    determine_core(master, instance_name, instancepath=instancepath)
    logger.message("Start Optimization (Classical Benders)")
    begin_time = time.time()
    # optimize the master with the callback according to the used Cut Selection Strategy
    master.optimize(lambda model, where: cb_mw_benders(model, where, dualsub, ret_object, begin_time))
    master._soltime = time.time() - begin_time
    # write output
    to_write = log_csv_line(instance_name, "mwb", master)
    with open(instancepath+"/"+instance_name+"/mwb.line", "w") as line_file:
        line_file.write(to_write)
    with open(logpath, "a") as my_file:
        my_file.write(to_write)


def MIS_benders(instance_name, al = 0.25, sr = False, logpath="log.csv", instancepath="PrepInstances/"):
    """
    Method to optimize a model with MIS Benders cuts
    """
    # parse (prepared) instance
    master, dualsub, ret_object = read_prep_instance(instance_name, sr = sr, al = al, instancepath=instancepath)
    modify_ds_for_mis(dualsub, ret_object, normalization = "standard")
    logger.message("Start Optimization (MIS Benders)")
    begin_time = time.time()
    # optimize the master with the callback according to the used Cut Selection Strategy
    master.optimize(lambda model, where: cb_mis_benders(model, where, dualsub, ret_object, begin_time))
    master._soltime = time.time() - begin_time
    # write output
    to_write = log_csv_line(instance_name, "mis", master)
    with open(instancepath+"/"+instance_name+"/mis.line", "w") as line_file:
        line_file.write(to_write)
    with open(logpath, "a") as my_file:
        my_file.write(to_write)


def facet_benders(instance_name, al = 0.25, sr = False, logpath="log.csv", instancepath="PrepInstances/"):
    """
    Method to optimize a model with Facet Benders cuts
    """
    # parse (prepared) instance
    master, dualsub, ret_object = read_prep_instance(instance_name, sr = sr, al = al, instancepath=instancepath)
    modify_ds_for_facet(dualsub, ret_object)
    logger.message("Start Optimization (Facet Benders)")
    begin_time = time.time()
    # optimize the master with the callback according to the used Cut Selection Strategy
    master.optimize(lambda model, where: cb_facet_benders(model, where, dualsub, ret_object, begin_time))
    master._soltime = time.time() - begin_time
    # write output
    to_write = log_csv_line(instance_name, "fcb", master)
    with open(instancepath+"/"+instance_name+"/fcb.line", "w") as line_file:
        line_file.write(to_write)
    with open(logpath, "a") as my_file:
        my_file.write(to_write)


def hybrid_benders(instance_name, al = 0.25, sr = False, logpath="log.csv", instancepath="PrepInstances/"):
    """
    Method to optimize a model with hybrid (first MIS, then OLS) Benders cuts
    """
    # parse (prepared) instance
    master, dualsub, ret_object = read_prep_instance(instance_name, sr = sr, al = al, instancepath=instancepath)
    modify_ds_for_mis(dualsub, ret_object)
    logger.message("Start Optimization (Hybrid Benders)")
    begin_time = time.time()
    # optimize the master with the callback according to the used Cut Selection Strategy
    master.optimize(lambda model, where: cb_hybrid_benders(model, where, dualsub, ret_object, begin_time))
    master._soltime = time.time() - begin_time
    # write output
    to_write = log_csv_line(instance_name, "hyb", master)
    with open(instancepath+"/"+instance_name+"/hyb.line", "w") as line_file:
        line_file.write(to_write)
    with open(logpath, "a") as my_file:
        my_file.write(to_write)


def optimal_line_shift_benders(instance_name, al = 0.25, sr = False, logpath="log.csv", instancepath="PrepInstances/"):
    """
    Method to optimize a model with OLS Benders cuts
    """
    # parse (prepared) instance
    master, dualsub, ret_object = read_prep_instance(instance_name, sr = sr, al = al, instancepath=instancepath)
    modify_ds_for_ols(dualsub, ret_object)
    logger.message("Start Optimization (OLS Benders)")
    begin_time = time.time()
    # optimize the master with the callback according to the used Cut Selection Strategy
    master.optimize(lambda model, where: cb_ols_benders(model, where, dualsub, ret_object, begin_time))
    master._soltime = time.time() - begin_time
    # write output
    to_write = log_csv_line(instance_name, "ols", master)
    with open(instancepath+"/"+instance_name+"/ols.line", "w") as line_file:
        line_file.write(to_write)
    with open(logpath, "a") as my_file:
        my_file.write(to_write)
    # print some statistics
    print(f"Calls: {master._count} d1: {master._d1count} i: {master._icount} f/md: {master._fcount}/{master._mdcount}")
    print(f"Time: d1 {master._d1time:4.4f} i {master._itime:4.4f} md {master._mdtime:4.4f} n {master._ntime:4.4f} a {master._atime:4.4f}")
    print(f"Maxalpha: {master._maxalpha} Giant_cut_norm_cuts: {master._giant_cut_norm} Alpha_at_max: {master._alpha_at_max} Depth_at_max: {master._depth_at_max}")


def vanilla_solve(instance_name, al = 0.25, sr = False, logpath="log.csv", instancepath="PrepInstances"):
    """
    Method to optimize a model without Benders Decomposition
    """
    # parse instance
    m = grb.read(f"{instancepath}/{instance_name}/{instance_name}.lp")
    logger.message("Start vanilla solve.")
    begin_time = time.time()
    # set some model parameters
    m.Params.TimeLimit = 3420.
    m.Params.Threads = 1
    m.Params.Seed = 1
    # optimize
    m.optimize()
    # some statistics
    m._count = 0
    m._soltime = time.time() - begin_time
    m._oc_time = 0.
    # write output
    to_write = log_csv_line(instance_name, "van", m)
    with open(instancepath+"/"+instance_name+"/van.line", "w") as line_file:
        line_file.write(to_write)
    with open(logpath, "a") as my_file:
        my_file.write(to_write)
    

if __name__ == "__main__":
    """
    Main script.
    """
    
    # handle command line arguments
    parser = argparse.ArgumentParser(description='Cut selection strategies - comparison.')
    parser.add_argument("instance")
    # https://stackoverflow.com/questions/14097061/easier-way-to-enable-verbose-logging
    parser.add_argument(
        '-m', '--mode',
        help="Optimization mode. \n 0: Classical Benders \n 1: Magnanti-Wong \n 2: MIS Benders \n 3: Facet Cuts \n 4: Optimal Line Shifts \n 5: MIP Solver \n 6: Hybrid \n 7: CPXR \n 8: CPX+OLSfeas",
        type=int,
        default=0,
    )
    parser.add_argument(
        '--logpath',
        help="Logpath. Default log.csv",
        type=str,
        default="log.csv",
    )
    parser.add_argument(
        '--instancepath',
        help="Instancepath. Default PrepInstances",
        type=str,
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
        '-a', '--approximation_level',
        help="Approximation Gap of initial solution heuristic",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        '-r', '--solve_relaxation',
        help="Solve the relaxation of the problem.",
        action="store_true",
    )
    parser.add_argument(
        '-s', '--random_seed',
        help="Random seed.",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()
    
    # set random seed
    random.seed(args.random_seed)
    
    # initialize logger (for command line output)
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
    
    # case distinctinction of args.mode;
    # apply the Benders method that corresponds to args.mode
    if args.mode == 0:
        classical_benders(args.instance, al = args.approximation_level, sr = args.solve_relaxation, logpath = args.logpath, instancepath = args.instancepath)
    elif args.mode == 1:
        magnanti_wong_benders(args.instance, al = args.approximation_level, sr = args.solve_relaxation, logpath = args.logpath, instancepath = args.instancepath)
    elif args.mode == 2:
        MIS_benders(args.instance, al = args.approximation_level, sr = args.solve_relaxation, logpath = args.logpath, instancepath = args.instancepath)
    elif args.mode == 3:
        facet_benders(args.instance, al = args.approximation_level, sr = args.solve_relaxation, logpath = args.logpath, instancepath = args.instancepath)
    elif args.mode == 4:
        optimal_line_shift_benders(args.instance, al = args.approximation_level, sr = args.solve_relaxation, logpath = args.logpath, instancepath = args.instancepath)
    elif args.mode == 5:
        vanilla_solve(args.instance, al = args.approximation_level, sr = args.solve_relaxation, logpath = args.logpath, instancepath = args.instancepath)
    elif args.mode == 6:
        hybrid_benders(args.instance, al = args.approximation_level, sr = args.solve_relaxation, logpath = args.logpath, instancepath = args.instancepath)
    elif args.mode == 7:
        cpx_benders(args.instance, al = args.approximation_level, sr = args.solve_relaxation, logpath = args.logpath, instancepath = args.instancepath)
    elif args.mode == 8:
        cpx_ols_benders(args.instance, al = args.approximation_level, sr = args.solve_relaxation, logpath = args.logpath, instancepath = args.instancepath)
    else:
        logger.warning("Invalid solution mode. Terminating")
        sys.exit()


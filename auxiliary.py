### Author Florian Roesel 08.09.2023

### AUXILIARY METHODS FOR compare_css.py
import gurobipy as grb
import logging
logger = logging.getLogger("__main__")


class model_attrs():
    def __init__(self):
        self.master_vars = 0.
        self.sub_vars = 0.
        self.master_cons = 0.
        self.mix_cons = 0.
        self.sub_cons = 0.
        self.all_sub_cons = 0.
        self.H = 0.
        self.f = 0.
        self.m = 0.


def create_approximate(instance_path, instance_name, al):
    """
    Reads an instance and calculates a solution that is (1+al)-optimal.
    Writes the .sol file in the instance directory.
    """
    m = grb.read(instance_path + instance_name + ".lp")
    m.Params.MIPGap = al
    m.Params.OutputFlag = 0.
    logger.warning("We have to calculate an appropriate approximate solution.")
    m.optimize()
    m.write(f"{instance_path}master_{al}.sol")


def create_relaxed_approximate(instance_path, instance_name, al):
    """
    Takes an instance and determines a solution of its LP-relaxation
    that has a value of (1+al)*optimal_value of the LP-relaxation.
    Writes the .sol file in the instance directory.
    """
    m = grb.read(instance_path + instance_name + ".lp")
    m.Params.MIPGap = 0.
    m.Params.OutputFlag = 0.
    m = m.relax()
    m.addVar(vtype = "B", name="dummy")
    m.update()
    logger.warning("We have to calculate an appropriate approximate solution.")
    m.optimize()
    m.addConstr(m.getObjective() >= m.objval * (1+al), name="objbound")
    m.update()
    m.optimize()
    m.write(f"{instance_path}master_{al}_r.sol")


def set_obs_attrs(master):
    """
    Takes Gurobi Model() master and initializes some attributes.
    """
    master._count = 0
    master._d1count = 0
    master._icount = 0
    master._mdcount = 0
    master._fcount = 0
    
    master._d1time = 0.
    master._itime = 0.
    master._mdtime = 0.
    master._ntime = 0.
    master._oc_time = 0.
    master._atime = 0.
    
    master._dualdensity = []
    master._cutdensity = []
    master._cutsteepness = []
    master._cutsteepnessnorm = []
    
    master._alpha_at_max = 0
    master._depth_at_max = 0
    master._giant_cut_norm = 0
    
    master._maxalpha = 0.
    master.update()


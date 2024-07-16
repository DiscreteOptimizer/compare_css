### Author Florian Roesel 08.09.2023

### A program to generate network design instances.
import networkx as nx
import gurobipy as grb
import argparse
import logging
logger = logging.getLogger("__main__")
import random
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os

"""
Methods to generate different classes of graphs:
Grid, Erdoes-Renyi, Random-Regular, Two graph classes that try to be similar to Fischetti et al. (2010)'s test instances 
"""
def random_grid(width, length, seed = 0):
    # Grid
    node_nbr = lambda node: f"{node[1] * width + node[0]}"
    graph = nx.grid_graph(dim = (range(length), range(width)))
    return graph


def random_random(n, seed = 0):
    # ERG
    graph = nx.fast_gnp_random_graph(n, 1./np.sqrt(n), seed=seed, directed=False)
    while not nx.is_connected(graph): # das ist natuerlich idiotisch....
        graph = nx.fast_gnp_random_graph(n, 1./np.sqrt(n), seed=None, directed=False)
    return graph


def random_rrg(n, seed = 0):
    # RRG
    degree = int(np.ceil(np.sqrt(n)))
    if degree % 2 == 1:
        degree -= 1
    graph = nx.random_regular_graph(degree, n, seed=seed)
    while not nx.is_connected(graph):
        graph = nx.random_regular_graph(degree, n, seed=None)
    return graph


def random_fisc(n, seed = 0):
    # Generate a graph with node degree between 2 and 6.
    def min_degree(graph):
        return min([element[1] for element in graph.degree()])
    def max_degree(graph):
        return max([element[1] for element in graph.degree()])
    graph = nx.Graph()
    for node in range(n):
        graph.add_node(node)
    while min_degree(graph) < 2:
        e1, e2 = random.sample(list(graph.nodes()), 2)
        if graph.degree(e1) <= 5 and graph.degree(e2) <= 5 and (e1, e2) not in list(graph.edges()) and (e2, e1) not in list(graph.edges()):
            graph.add_edge(e1, e2)
    return graph


def random_fisc_2(n, seed = 0):
    # Start with 4-RRG, remove edges, add edges.
    graph = nx.random_regular_graph(4, n, seed=seed)
    edges_to_delete = []
    for e1, e2 in graph.edges():
        if random.random() <= 0.06:
            edges_to_delete.append((e1, e2))
    for e1, e2 in edges_to_delete:
        graph.remove_edge(e1, e2)
    sample_list = list(range(n))
    for i in range(13):
        graph.add_edge(*random.sample(sample_list, 2))
    return graph


def generate_random_graph_mcfnd_instance(graph, commodities = 50, fairness = False, width = 0, cap_costs = "constant", flow_costs = "constant"):
    """
    Take a graph, commodity number, an generate a MCF-NWD instance
    """
    # generate the MIP instance of a graph
    node_nbr = lambda node: node
    if width:
        node_nbr = lambda node: f"{node[1] * width + node[0]}"
    
    # 0.0 commodity weights, costs, cap_costs
    weight = []
    flow_list = []
    # flow costs according to flow_costs attribute
    for com in range(commodities):
        weight.append(np.random.randint(5, 10))
        if flow_costs == "constant":
            flow_list.append(1.)
        elif flow_costs == "low":
            flow_list.append(0.1)
        elif flow_costs == "zero":
            flow_list.append(0.)
        elif flow_costs == "random":
            flow_list.append(0.7 + np.random.randint(7) * 0.1)
        else:
            logger.warning(f"Invalid flow cost option {flow_costs}. Applying constant.")
            flow_list.append(1.)
    
    model = grb.Model()
    lb_vars = {}
    cap_vars = {}
    flow_vars = {}
    # 2. iterate over arcs; one cap variable per arc...
    for e1, e2 in graph.edges():
        if cap_costs == "random":
            cap_cost = np.random.randint(3, 8)
        else:
            cap_cost = 5.
        cap_vars[f"{e1}_{e2}"] = model.addVar(vtype = "I", obj = cap_cost, name = f"y_{node_nbr(e1)}_{node_nbr(e2)}")
        # 2.1 iterate over commodities and generate two flow variables per arc
        flow_vars[f"{e1}_{e2}"] = []
        flow_vars[f"{e2}_{e1}"] = []
        for com in range(commodities):
            flow_vars[f"{e1}_{e2}"].append(model.addVar(vtype = "C", obj = flow_list[com], name = f"x_{node_nbr(e1)}_{node_nbr(e2)}_{com}"))
            flow_vars[f"{e2}_{e1}"].append(model.addVar(vtype = "C", obj = flow_list[com], name = f"x_{node_nbr(e2)}_{node_nbr(e1)}_{com}"))
    model.update()
    # 3. iterate over arcs and add capacity constraints
    for e1, e2 in graph.edges():
        model.addConstr(sum(weight[com] * var for com, var in enumerate(flow_vars[f"{e1}_{e2}"])) + sum(weight[com] * var for com, var in enumerate(flow_vars[f"{e2}_{e1}"])) <= 20 * cap_vars[f"{e1}_{e2}"], name = f"c_{node_nbr(e1)}_{node_nbr(e2)}")
    # 4. iterate over nodes of the graph and add flow conservation constraints
    for com in range(commodities):
        source, target = random.sample(list(graph.nodes()), 2)
        for node in graph.nodes():
            if node == source: rhs = 1.
            elif node == target: rhs = -1.
            else: rhs = 0.
            model.addConstr(sum(flow_vars[f"{e1}_{e2}"][com] - flow_vars[f"{e2}_{e1}"][com] for e1, e2 in graph.edges(node)) == rhs, name = f"f_{node_nbr(node)}_{com}")
    
    """
    # Dead code, commented out. Not used in calculations.
    # partition constraints
    if fairness:
        for e1, e2, edge_data in graph.edges(data=True):
            edge_data["p"] = np.random.randint(5)
        lb_vars["lb"] = model.addVar(vtype = "I", obj = 0., name = "partition_cap")
        model.update()
        for p in range(5):
            model.addConstr(1.0 * lb_vars["lb"] <= sum(int(edge_data["p"] == p) * cap_vars[f"{e1}_{e2}"] for e1, e2, edge_data in graph.edges(data=True)), name = f"lower_partition_{p}")
            model.addConstr(1.1 * lb_vars["lb"] >= sum(int(edge_data["p"] == p) * cap_vars[f"{e1}_{e2}"] for e1, e2, edge_data in graph.edges(data=True)), name = f"upper_partition_{p}")
    """
    model.update()
    return model
    


if __name__ == "__main__":
    """
    Main script.
    """
    parser = argparse.ArgumentParser(description='Generate a grid MC flow network design instance.')
    parser.add_argument("-m", "--mode",
                        type=int,
                        default=1,
                        help="1: grid, 2: ERG, 3: random regular, 4: fischetti-like, 5: fischetti-like2"
                        )
    parser.add_argument("-i", "--instancepath",
                        help="Which folder to put the instance to.",
                        default="NWD_Instances",
                        )
    parser.add_argument("-w", "--width",
                        type=int,
                        default=5,
                        )
    parser.add_argument("-l", "--length",
                        type=int,
                        default=5,
                        )
    parser.add_argument("-c", "--commodities",
                        type=int,
                        default=100,
                        )
    parser.add_argument("-n", "--number",
                        type=int,
                        default=0,
                        )
    parser.add_argument("-f", "--fairness",
                        action="store_true",
                        default=False,
                        )
    parser.add_argument("--cap_costs",
                        default="constant",
                        help = "Cap cost option. constant random",
                        )
    parser.add_argument("--flow_costs",
                        default="constant",
                        help = "Flow cost option. constant low zero random",
                        )
    
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
        type=int,
        default=123456,
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
    if args.mode == 1:
        graph = random_grid(args.width, args.length, seed=args.random_seed)
        model = generate_random_graph_mcfnd_instance(graph, args.commodities, args.fairness, args.width, args.cap_costs, args.flow_costs)
    elif args.mode == 2:
        graph = random_random(args.width*args.length, seed=args.random_seed)
        model = generate_random_graph_mcfnd_instance(graph, args.commodities, args.fairness, 0, args.cap_costs, args.flow_costs)
    elif args.mode == 3:
        graph = random_rrg(args.width*args.length, seed=args.random_seed)
        model = generate_random_graph_mcfnd_instance(graph, args.commodities, args.fairness, 0,  args.cap_costs, args.flow_costs)
    elif args.mode == 4:
        graph = random_fisc(args.width*args.length, seed=args.random_seed)
        model = generate_random_graph_mcfnd_instance(graph, args.commodities, args.fairness, 0,  args.cap_costs, args.flow_costs)
    elif args.mode == 5:
        graph = random_fisc_2(args.width*args.length, seed=args.random_seed)
        model = generate_random_graph_mcfnd_instance(graph, args.commodities, args.fairness, 0, args.cap_costs, args.flow_costs)
    else:
        logger.warning(f"Invalid mode {args.mode}. Type an int from 1 to 5.")
        sys.exit()
    if os.path.exists(f"{args.instancepath}"):
        pass
    else:
        os.mkdir(f"{args.instancepath}")
    model.write(f"{args.instancepath}/g{args.mode}_{args.width}_{args.length}_o_{args.number}.lp")


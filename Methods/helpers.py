import pyomo.environ as pyo
from Models.ModelType import ModelType
from Models.LPModel import LPModel
from Models.MILPModel import MILPModel
from Models.SiteModel import SiteModel
from Models.PortfolioModel import PortfolioModel
import numpy as np


# -- Initialize Solver --
def build_solver(solver_name):
    return pyo.SolverFactory(solver_name)


# -- Build model --
def build_model(formulation):
    if formulation == "LPModel":
        m = LPModel(type=ModelType.LP_Rafal)
    if formulation == "MILPModel":
        m = MILPModel(type=ModelType.MILP)
    if formulation == "SiteModel":
        m = SiteModel(type=ModelType.ALR_Site)
    if formulation == "PortfolioModel":
        m = PortfolioModel(type=ModelType.ALR_Portfolio)
    return m


def extract_from_locations(location_instance):
    """Given a solved location_instance, the function extracts the necessary information to be fed into the portfolio subproblem before solving"""
    dualsT = []
    exports_S_TL = np.zeros((int(pyo.value(location_instance.N_t)), int(pyo.value(location_instance.N_l))))
    imports_S_TL = np.zeros((int(pyo.value(location_instance.N_t)), int(pyo.value(location_instance.N_l))))
    for t in location_instance.T:
        # print(f"Time: {t} -- ",pyo.value(location_instance.dual[t]))
        dualsT.append(pyo.value(location_instance.dual[t]))
        for l in location_instance.L:
            exports_S_TL[t-1,l-1] = max(0,pyo.value(location_instance.e_S[t,l])) #BUG somtimes this NonNegativeReal is negative (-1E-8)... huh
            if pyo.value(location_instance.e_S[t,l]) < 0:
                print("BUG")
            imports_S_TL[t-1,l-1] = pyo.value(location_instance.i_S[t,l])
    
    return dualsT, exports_S_TL, imports_S_TL


def feed_to_portfolio(portfolio_instnace, dualsT, exports_S_TL, imports_S_TL):
    """Function feeds the necessary paramter data to portfolio before being solved. """
    for t in portfolio_instnace.T:
        # portfolio_instnace.dual[t] = dualsT[t-1] # We do not update the duals here
        for l in portfolio_instnace.L:
            portfolio_instnace.e_S[t,l] = exports_S_TL[t-1,l-1]
            portfolio_instnace.i_S[t,l] = imports_S_TL[t-1,l-1]

    return portfolio_instnace


def extract_from_portfolio(portfolio_instance):
    dualsT = []
    exports_G_T = []
    imports_G_T = []

    for t in portfolio_instance.T:
        dualsT.append(pyo.value(portfolio_instance.dual[t]))
        exports_G_T.append(pyo.value(portfolio_instance.e_G[t]))
        imports_G_T.append(pyo.value(portfolio_instance.e_G[t]))

    return dualsT, exports_G_T, imports_G_T


def feed_to_locations(locations_instance, dualsT, exports_G_T, imports_G_TL):
    for t in locations_instance.T:
        # locations_instance.dual[t] = dualsT[t-1] # We do not update the duals here 
        locations_instance.e_G[t] = exports_G_T[t-1]
        locations_instance.i_G[t] = imports_G_TL[t-1]
    return locations_instance

def update_the_duals(locations_instance, portfolio_instance):
    dualsT = []
    primal_residualT = []

    for t in locations_instance.T:
        primal_residual = (pyo.value(portfolio_instance.commitment_i[t])+ pyo.value(portfolio_instance.i_G[t]) + sum(pyo.value(locations_instance.e_S[t,l]) for l in locations_instance.L) - pyo.value(portfolio_instance.commitment_e[t]) - pyo.value(portfolio_instance.e_G[t]) - sum(pyo.value(locations_instance.i_S[t,l]) for l in locations_instance.L))
        mama =  pyo.value(locations_instance.dual[t]) + pyo.value(locations_instance.dualgamma[t])*primal_residual
        
        dualsT.append(mama)
        primal_residualT.append(primal_residual)

        locations_instance.dual[t] = mama
        portfolio_instance.dual[t] = mama

        # print(f"time {t} --", primal_residual)

    return locations_instance, portfolio_instance, np.array(dualsT), np.array(primal_residualT)
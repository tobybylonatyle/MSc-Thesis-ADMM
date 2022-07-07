import pyomo.environ as pyo
from Models.ModelType import ModelType
from Models.LPModel import LPModel
from Models.MILPModel import MILPModel
from Models.SiteModel import SiteModel
from Models.PortfolioModel import PortfolioModel


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






from enum import Enum


class ModelType(Enum):
    MILP = 0 #MILP
    LP_relaxation = 1 # Lagrange Relaxation
    SP_portfolio = 2 #Sub problem Portfolio
    SP_location = 3 # Sub problem Site
    LP_Rafal = 4 # 
    ALR = 5 # Augument Lagrangian Relaxation for use with ADMM
    LR = 6 #Lagrangian relaxationof energy balance portfolio constraint
    ALR_Portfolio = 7
    ALR_Site = 8
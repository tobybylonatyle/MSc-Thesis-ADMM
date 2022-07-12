import Methods.helpers as helpers
import time
import pyomo.environ as pyo
import numpy as np


SOLVER_NAME = 'gurobi'
INSTANCE_SIZE = 10
EQUAL_PRICES = False
MAX_ITER = 100
TOLERANCE = 1

def solve_MILP():
    print("> Solving MILP")
    solver = helpers.build_solver(SOLVER_NAME)
    model = helpers.build_model('MILPModel')

    instance = model.build_instance(instance_size=INSTANCE_SIZE, equal_prices=EQUAL_PRICES)

    start_time = time.perf_counter()
    result = solver.solve(instance)
    print(time.perf_counter()- start_time)


    print(pyo.value(instance.Objective_Cost))

def solve_LP():
    print("> Solving LP")
    solver = helpers.build_solver(SOLVER_NAME)
    model = helpers.build_model('LPModel')

    instance = model.build_instance(instance_size=INSTANCE_SIZE, equal_prices=EQUAL_PRICES)

    start_time = time.perf_counter()
    result = solver.solve(instance)
    print(time.perf_counter()- start_time)


    print(pyo.value(instance.Objective_Cost))

def solve_ADMM():
    print("> Solving ADMM")
    solver = helpers.build_solver(SOLVER_NAME)
 
    # -- Create |L| +1 decomposed subproblems -- 
    location_models = {}
    location_instances = {}
    for l in range(INSTANCE_SIZE):
        print(f"   Creating Subproblem for Location {l}")
        location_models[l] = helpers.build_model("SiteModel")
        location_instances[l] = location_models[l].build_instance(instance_size=INSTANCE_SIZE, equal_prices=EQUAL_PRICES)

        for t in location_instances[l].T:
            location_instances[l].dualgamma[t] = 0


    print("HERE")
    portfolio_model = helpers.build_model("PortfolioModel")
    portfolio_instnace = portfolio_model.build_instance(instance_size=INSTANCE_SIZE, equal_prices=EQUAL_PRICES)
    print("THERE")

    # -- END : Create |L| +1 decomposed subproblems -- 

    # -- ADMM PART SUPER COOL LETS GO -- 
    # for l in range(INSTANCE_SIZE):
    #     for t in location_models[l].T:
    #         location_models[l].i_G[t] = 0
    #         location_models[l].e_G[t] = 0

            
    # ITER_MAX = 12
    # while(iter <= ITER_MAX):
    #     result = solver.solve(location_instances[0])

    #     solver.solve()
        




    # TODO: Find a way of setting the dual
    # TODO: Store and Feed data to models for iterations
    # TODO: Find a way of dynamically updating the dualgamma

    result = solver.solve(location_instances[0])
    print(result)
    # for t in location_instances[0].T:
    #     print(pyo.value(location_instances[1].dual[t]))

    


def solve_RFL():
    """Sequential ADMM, first solve 1st subproblem, feed those vars as params into 2nd subproblem, 
    feed 2nd subproblem vars into 1st subproblem as params then update duals, and repeat """

    #TODO understand how updating duals works in this specific problem!!!!
    #TODO Try to make SiteModel and PortfolioModel LPs!!!!!! 


    computational_data = {'obj_cost': [], 'dualized_constraint_value' : [], 'augmentation_value': [], 'dualsT': [], 'primal_residualsT': [], 'dualgammasT' : [] }

    print(" > Solving RFL")
    solver = helpers.build_solver(SOLVER_NAME)

    locations_model = helpers.build_model("SiteModel")
    locations_instance = locations_model.build_instance(instance_size=INSTANCE_SIZE, equal_prices=EQUAL_PRICES)

    portfolio_model = helpers.build_model("PortfolioModel")
    portfolio_instnace = portfolio_model.build_instance(instance_size=INSTANCE_SIZE, equal_prices=EQUAL_PRICES)

    # Intialize for the very first solve 
    for t in locations_instance.T:
        locations_instance.dualgamma[t] = -25 # Fixed dualgamma for now
        locations_instance.i_G[t] = 0
        locations_instance.e_G[t] = 0

    for t in portfolio_instnace.T:
        portfolio_instnace.dualgamma[t] = -25

    iter = 0

    # SEQUENTIAL ADMM 
    while(iter < MAX_ITER):
        print(f"iteration {iter}")

        
        result_p                            = solver.solve(locations_instance) # Solve locations subproblem
        exports_S_TL, imports_S_TL          = helpers.extract_from_locations(locations_instance) # Extract data from locations supbroblem 
        portfolio_instance                  = helpers.feed_to_portfolio(portfolio_instnace, exports_S_TL, imports_S_TL) # Feed locations subproblem data into portfolio subproblem
        result_l                            = solver.solve(portfolio_instance) # Solve portfolio subproblem
        exports_G_T, imports_G_T            = helpers.extract_from_portfolio(portfolio_instance) # Extract data from portfolio subproblem
        locations_instance                  = helpers.feed_to_locations(locations_instance, exports_G_T, imports_G_T) # Feed portfolio subproblem data into locations subproblem
        
        locations_instance, portfolio_instance, dualsT, primal_residualsT, dualgammasT = helpers.update_the_duals(locations_instance, portfolio_instance) # Update the dual variables


        objective_cost_original, dualized_constraint_value, augmentation_value =  helpers.calculate_obj_cost(locations_instance, portfolio_instance)
     

        # Store computational Results
        computational_data['obj_cost'].append(objective_cost_original)
        computational_data['dualized_constraint_value'].append(dualized_constraint_value)
        computational_data['augmentation_value'].append(augmentation_value)
        computational_data['dualsT'].append(dualsT)
        computational_data['primal_residualsT'].append(primal_residualsT)
        computational_data['dualgammasT'].append(dualgammasT)





        #TODO Termination Condition

        iter += 1 
    # Quickly Process Computational Data: lists of lists are covnerted into a numpy matrix. 
    # Indicies are in order (iteration, time period)
    computational_data['dualsT'] = np.stack(computational_data['dualsT'],axis=0)
    computational_data['primal_residualsT'] = np.stack(computational_data['primal_residualsT'],axis=0)
    computational_data['dualgammasT'] = np.stack(computational_data['dualgammasT'],axis=0)


    return computational_data

def solve_MSL():
    """Parallel ADMM,  feed portolio_instance and locations_instance simulatniously, then """
    pass 


if __name__ == '__main__':
    # solve_LP()
    # solve_MILP()
    # solve_ADMM() # This is where each location is also a subproblem ie |L|+1 subproblems
    solve_RFL() # Just two subproblems, one for portfolio, onr for locations:::: Sequentially 
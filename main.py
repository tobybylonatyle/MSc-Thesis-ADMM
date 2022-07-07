import Methods.helpers as helpers
import time
import pyomo.environ as pyo


SOLVER_NAME = 'gurobi'
INSTANCE_SIZE = 10
EQUAL_PRICES = False

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

    




if __name__ == '__main__':
    solve_LP()
    solve_MILP()
    solve_ADMM()
import matplotlib.pyplot as plt
import _pickle as CPickle
from os import listdir

def save_run(results):
    i = len(listdir("Results/"))
    with open(f'Results/results{i}', 'wb') as f:
        CPickle.dump(results, f)
    print("done")

def open_run(path):
    with open(r"Results/"+path, 'rb') as input_file:
        temp = CPickle.load(input_file)
    return temp


def check_complementarities_locations_subproblem(decision_vars):
    for iter in decision_vars.keys():
        T, B, L = decision_vars[iter]['charge_TBL'].shape
        for t in range(T):
            if (decision_vars[iter]['e_G'][t]*decision_vars[iter]['i_G'][t] != 0):
                print(f"Import Export, iteration {iter}, time {t}")

            for l in range(L):
                if (decision_vars[iter]['e_S'][t,l]*decision_vars[iter]['i_S'][t,l] != 0):
                    print(f"send/recieve, iteration {iter},location {l}, time {t} ")
            
                for b in range(B):
                    if (decision_vars[iter]['charge_TBL'][t,b,l]*decision_vars[iter]['discharge_TBL'][t,b,l] != 0):
                        print(f"charge/discharge, iter {iter}, location {l}, time {t}, battery {b}")



def plot_results_behaviour(results):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12))

    tperiods = results['dualsT'].shape[1]
    iters = results['dualsT'].shape[0]
    dualgamma = results['dualgammasT'][1,0]
    
    fig.suptitle(f"Time Perios: {tperiods}, Iterations: {iters}, DualGamma: {dualgamma}")
    ax[0,0].plot(results['obj_cost'][1:])
    ax[0,0].set(xlabel="iteration", ylabel = 'objective value original')
    
    ax[1,1].plot(results['min_obj'][1:])
    ax[1,1].set(xlabel="iteration", ylabel = 'minimization objective')

    for i in range(results['dualsT'].shape[1]):
        ax[0,1].plot(results['dualsT'][1:,i])
        ax[0,1].set(xlabel="iteration",ylabel='dual values')

        ax[1,0].plot(results['primal_residualsT'][1:,i])
        ax[1,0].set(xlabel="iteration", ylabel='primal residual')



    plt.show()
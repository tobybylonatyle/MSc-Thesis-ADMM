import matplotlib.pyplot as plt


def plot_computational_behaviour(computational_results):
    """omits the zeroth iteration which is just a bad initialization...."""
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12))

    tperiods = computational_results['dualsT'].shape[1]
    iters = computational_results['dualsT'].shape[0]
    dualgamma = computational_results['dualgammasT'][1,0]
    
    fig.suptitle(f"Time Perios: {tperiods}, Iterations: {iters}, DualGamma: {dualgamma}")
    ax[0,0].plot(computational_results['obj_cost'][1:])
    ax[0,0].set(xlabel="iteration", ylabel = 'objective value original')
    
    ax[1,1].plot(computational_results['min_obj'][1:])
    ax[1,1].set(xlabel="iteration", ylabel = 'minimization objective')

    for i in range(computational_results['dualsT'].shape[1]):
        ax[0,1].plot(computational_results['dualsT'][1:,i])
        ax[0,1].set(xlabel="iteration",ylabel='dual values')

        ax[1,0].plot(computational_results['primal_residualsT'][1:,i])
        ax[1,0].set(xlabel="iteration", ylabel='primal residual')

    plt.show()

def plot_duality_gap(computational_results):
    plt.plot(computational_results['min_obj'][1:])
    plt.plot(computational_results['obj_cost'][1:])

    plt.show()
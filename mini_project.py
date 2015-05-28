from matplotlib import pyplot as plt
import numpy as np


default_number_neurons = 100.0
default_connection_probability = 0.30
default_synaptic_weights = 0.2

range_number_neurons = np.arange(100, 200, 2)
range_connection_probabilities = np.arange(0.3, 0.61, 0.015)
range_synaptic_weights = np.arange(0.2, 0.4, 0.01)

default_synaptic_time_constant = 10  # ms
range_tau_syn = np.arange(10.0, 20.0, 0.25)


def run_sim(number_neurons=default_number_neurons,
            connection_probability=default_connection_probability,
            synaptic_weights=default_synaptic_weights,
            synaptic_time_constant=default_synaptic_time_constant,
            tend=300):
    from brian.units import mvolt, msecond, namp, Mohm
    import brian

    El = 0 * mvolt
    tau_m = 30 * msecond
    tau_syn = synaptic_time_constant * msecond
    R = 20 * Mohm
    v_threshold = 30 * mvolt
    v_reset = 0 * mvolt
    tau_refractory = 4 * msecond

    eqs = brian.Equations('''
    dv/dt = (-(v - El) + R*I)/tau_m : volt
    I = I_syn + I_stim : amp
    dI_syn/dt = -I_syn/tau_syn : amp
    I_stim : amp
    ''')

    external_current = np.zeros(tend)
    external_current[np.arange(0, 100)] = 5 * namp

    group = brian.NeuronGroup(
        model=eqs,
        N=number_neurons, threshold=v_threshold, reset=v_reset, refractory=tau_refractory)

    group.I_stim = brian.TimedArray(external_current, dt=1*msecond)

    connections = brian.Connection(group, group, 'I_syn')
    connections.connect_random(sparseness=connection_probability, weight=synaptic_weights*namp)

    spike_monitor = brian.SpikeMonitor(group)
    population_rate_monitor = brian.PopulationRateMonitor(group, bin=10*msecond)

    brian.reinit()
    brian.run(tend * msecond)

    return spike_monitor, population_rate_monitor


def calculate_late_response(population_rate_monitor, t0=0.250, t1=0.300):
    return np.mean(
        population_rate_monitor.rate[
            (population_rate_monitor.times >= t0) &
            (population_rate_monitor.times <= t1)])


def plot_post_synaptic_current(k, tau, t0, t1):
    t = np.arange(t0, t1, 0.1)
    psc = k * np.exp(-t / tau)
    psc[t < 0] = 0

    plt.figure(figsize=(4, 2))
    plt.title('Post-Synaptic Current')
    plt.ylabel('Current (nA)')
    plt.xlabel('Time (ms)')
    plt.xlim(xmin=t0, xmax=t1)
    plt.tight_layout()
    plt.plot(t, psc)
    plt.savefig('results/post_synaptic_current.png')


def exercise_1():
    from brian.plotting import raster_plot

    spike_monitor, population_rate_monitor = run_sim()

    #############################################

    plt.figure(figsize=(3.0, 2.5))
    plt.xlim(xmin=0, xmax=300)
    raster_plot(spike_monitor, title='Spike Raster', showplot=False)
    plt.tight_layout()
    plt.savefig('results/raster_plot.png')

    #############################################

    plt.figure(figsize=(3.0, 2.5))
    plt.title('Population Rate')
    plt.ylabel('Rate (Hz)')
    plt.xlabel('Time (ms)')
    plt.xlim(xmin=0, xmax=300)
    plt.tight_layout()
    plt.plot(population_rate_monitor.times * 1000, population_rate_monitor.rate)
    plt.savefig('results/population_rate.png')

    #############################################

    late_response = calculate_late_response(population_rate_monitor)
    print 'late response:', late_response

    #############################################

    plot_post_synaptic_current(0.2, 10, -10, 100)


def experiment_connection_probabilities():
    nn = np.ones(range_connection_probabilities.size) * default_number_neurons
    sw = np.ones(range_connection_probabilities.size) * default_synaptic_weights
    ts = np.ones(range_connection_probabilities.size) * default_synaptic_time_constant

    late_responses = calculate_late_responses_experiment(nn, range_connection_probabilities, sw, ts)
    return range_connection_probabilities, late_responses


def experiment_synaptic_weights():
    nn = np.ones(range_synaptic_weights.size) * default_number_neurons
    cp = np.ones(range_synaptic_weights.size) * default_connection_probability
    ts = np.ones(range_synaptic_weights.size) * default_synaptic_time_constant

    late_responses = calculate_late_responses_experiment(nn, cp, range_synaptic_weights, ts)
    return range_synaptic_weights, late_responses


def experiment_number_neurons():
    cp = np.ones(range_number_neurons.size) * default_connection_probability
    sw = np.ones(range_number_neurons.size) * default_synaptic_weights
    ts = np.ones(range_number_neurons.size) * default_synaptic_time_constant

    late_responses = calculate_late_responses_experiment(range_number_neurons, cp, sw, ts)
    return range_number_neurons, late_responses


def experiment_synaptic_time_constant():
    nn = np.ones(range_tau_syn.size) * default_number_neurons
    cp = np.ones(range_tau_syn.size) * default_connection_probability
    sw = np.ones(range_tau_syn.size) * default_synaptic_weights
    late_responses = calculate_late_responses_experiment(nn, cp, sw, synaptic_time_constant=range_tau_syn)

    return range_tau_syn, late_responses


def calculate_late_responses_experiment(number_neurons, connection_probability, synaptic_weights, synaptic_time_constant):
    late_responses = []

    for nn, cp, sw, tc in zip(number_neurons, connection_probability, synaptic_weights, synaptic_time_constant):
        print nn, cp, sw, tc
        spike_monitor, population_rate_monitor = run_sim(
            number_neurons=nn,
            connection_probability=cp,
            synaptic_weights=sw,
            synaptic_time_constant=tc)
        late_responses.append(calculate_late_response(population_rate_monitor))

    return np.array(late_responses)


def plot_exp(variable, late_responses, xlabel, ylabel):
    plt.figure(figsize=(5, 2.5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.plot(variable, late_responses, '-')
    plt.plot(variable, late_responses, 'o')
    plt.savefig('results/%s_by_%s.png' %
                (ylabel.lower().replace(' ', '_'),
                 xlabel.lower().replace(' ', '_')))


def exercise_2():
    cp, late_responses = experiment_connection_probabilities()
    plot_exp(cp, late_responses, 'Connection Probability', 'Late Response (Hz)')

    #############################################

    tend = 3000
    plt.figure(figsize=(5, 3))

    for cp in [0.36, 0.37, 0.45, 0.50]:
        _, monitor = run_sim(connection_probability=cp, tend=tend)
        plt.plot(monitor.times * 1000, monitor.rate, label='C. Prob. %d%%' % (cp * 100))

    plt.ylabel('Rate (Hz)')
    plt.xlabel('Time (ms)')
    plt.xlim(xmin=0, xmax=tend)
    plt.tight_layout()
    plt.legend(prop={'size': 8})
    plt.savefig('results/population_rate_compare_conn_prob.png')


def exercise_3():
    sw, late_responses = experiment_synaptic_weights()
    plot_exp(sw, late_responses, 'Synaptic weights (nA)', 'Late Response (Hz)')


def exercise_4():
    nn, late_responses = experiment_number_neurons()
    plot_exp(nn, late_responses, 'Number of Neurons', 'Late Response (Hz)')


def compute_effective_coefficient(number_neurons, connection_probabilities, synaptic_weights):
    tau = 0.01  # seconds
    k = 1.0

    post_synaptic_charge = tau * k
    return synaptic_weights * number_neurons * connection_probabilities * post_synaptic_charge


def get_intersections(f0, f1, x0, x1):
    from scipy.optimize import fsolve

    def find_intersection(initial):
        return fsolve(lambda _x: f0(_x) - f1(_x), initial, full_output=True)

    intersections = []
    for x in np.arange(x0, x1, 0.1):
        sol, _, found, _ = find_intersection(x)
        if found == 1:
            intersections.append(sol[0])

    rounded = np.unique(np.around(intersections))
    return np.array([rounded, f0(rounded)])


def plot_activity_by_synaptic_current_from_average_synaptic_current(gain):
    syn_current = gain[0]  # nano-amps

    for cp in [default_connection_probability, 0.365, 0.45]:
        effective_coefficient = compute_effective_coefficient(default_number_neurons,
                                                              cp,
                                                              default_synaptic_weights)
        activity = syn_current / effective_coefficient
        line = plt.plot(
            syn_current, activity,
            label='From Syn Current (EC: %.3f)' % effective_coefficient)

        intersections = get_intersections(lambda curr: curr / effective_coefficient,
                                          lambda curr: np.interp(curr, gain[0], gain[1]),
                                          gain[0].min(), gain[1].max())

        print 'coefficient:', effective_coefficient
        for i in range(intersections.shape[1]):
            print 'intersection:', intersections[0][i], intersections[1][i]

        plt.plot(intersections[0], intersections[1], 'o', color=line[0].get_color())


def exercise_5():
    plt.figure(figsize=(4.0, 2.5))
    gain_filename = 'gain.txt'
    gain = np.loadtxt(gain_filename)
    plt.plot(gain[0, :], gain[1, :], label='From Gain')
    plot_activity_by_synaptic_current_from_average_synaptic_current(gain)
    plt.ylabel('Activity (Hz)')
    plt.xlabel('Synaptic Current (nA)')
    plt.ylim(ymin=-10, ymax=gain[1].max()*1.05)
    plt.legend(loc='lower right', prop={'size': 8})
    plt.tight_layout()
    plt.savefig('results/activity_by_synaptic_current')

    #############################################

    plt.figure(figsize=(5.0, 2.5))
    max_coef = 0

    print 'experiment_connection_probabilities'
    connection_probabilities, late_responses = experiment_connection_probabilities()
    coefficient = compute_effective_coefficient(
        default_number_neurons, connection_probabilities, default_synaptic_weights)
    plt.plot(coefficient, late_responses, label='Connection Probability')
    max_coef = max(np.max(coefficient), max_coef)

    print 'experiment_synaptic_weights'
    synaptic_weights, late_responses = experiment_synaptic_weights()
    coefficient = compute_effective_coefficient(
        default_number_neurons, default_connection_probability, synaptic_weights)
    plt.plot(coefficient, late_responses, label='Synaptic Weights')
    max_coef = max(np.max(coefficient), max_coef)

    print 'experiment_number_neurons'
    number_neurons, late_responses = experiment_number_neurons()
    coefficient = compute_effective_coefficient(
        number_neurons, default_connection_probability, default_synaptic_weights)
    plt.plot(coefficient, late_responses, label='Number of Neurons')
    max_coef = max(np.max(coefficient), max_coef)

    plt.xlabel('Effective Coefficient')
    plt.ylabel('Late Response (Hz)')
    plt.tight_layout()
    plt.xlim(xmax=max_coef)
    plt.legend(loc='lower right', prop={'size': 8})
    plt.savefig('results/late_response_(hz)_by_effective_coefficient.png')

    #############################################

    tc, late_responses = experiment_synaptic_time_constant()
    plot_exp(tc, late_responses, 'Synaptic Time Constant (ms)', 'Late Response (Hz)')


def main():
    '''run all exercises'''
    exercises = [
        exercise_1,
        exercise_2,
        exercise_3,
        exercise_4,
        exercise_5
    ]

    for i, excercise in enumerate(exercises):
        print 'excercise', i + 1
        excercise()

    print 'done'


if __name__ == '__main__':
    main()

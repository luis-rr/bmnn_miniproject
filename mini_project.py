from matplotlib import pyplot as plt
import numpy as np


default_number_neurons = 100.0
default_connection_probabilities = 0.30
default_synaptic_weights = 0.2


def run_sim(number_neurons, connection_probability, synaptic_weights, tend=300, external_current_strength=5):
    from brian.units import mvolt, msecond, namp, Mohm
    import brian

    El = 0 * mvolt
    tau_m = 30 * msecond
    tau_syn = 10 * msecond
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
    external_current[np.arange(0, 100)] = external_current_strength * namp

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


def calculate_late_responses_experiment(number_neurons, connection_probability, synaptic_weights):
    late_responses = []

    for nn, cp, sw in zip(number_neurons, connection_probability, synaptic_weights):
        print nn, cp, sw
        spike_monitor, population_rate_monitor = run_sim(nn, cp, sw)
        late_responses.append(calculate_late_response(population_rate_monitor))

    return np.array(late_responses)


def plot_post_synaptic_current(k, tau, t0, t1):
    t = np.arange(t0, t1, 0.1)
    psc = k * np.exp(-t / tau)
    psc[t < 0] = 0

    f = plt.figure(figsize=(4, 2))
    plt.title('Post-Synaptic Current')
    plt.ylabel('Current (nA)')
    plt.xlabel('Time (ms)')
    plt.xlim(xmin=t0, xmax=t1)
    f.tight_layout()
    plt.plot(t, psc)
    plt.savefig('results/post_synaptic_current.png')


def plot_population_rate(population_rate_monitor):
    f = plt.figure(figsize=(4, 3))
    plt.title('Population Rate')
    plt.ylabel('Rate (Hz)')
    plt.xlabel('Time (ms)')
    plt.xlim(xmin=0, xmax=300)
    f.tight_layout()
    plt.plot(population_rate_monitor.times * 1000, population_rate_monitor.rate)
    plt.savefig('results/population_rate.png')


def exercise_1():
    from brian.plotting import raster_plot

    spike_monitor, population_rate_monitor = run_sim(
        default_number_neurons,
        default_connection_probabilities,
        default_synaptic_weights
    )

    print 'n spikes:', spike_monitor.nspikes
    f = plt.figure(figsize=(4, 3))
    plt.xlim(xmin=0, xmax=300)
    raster_plot(spike_monitor, title='Spike Raster', showplot=False)
    f.tight_layout()
    plt.savefig('results/raster_plot.png')

    plot_population_rate(population_rate_monitor)

    late_response = calculate_late_response(population_rate_monitor)
    print 'late response:', late_response

    plot_post_synaptic_current(0.2, 10, -10, 100)


# range_synaptic_weights = np.arange(0.2, 0.4, 0.01)
# range_number_neurons = np.arange(100, 200, 2)
# range_connection_probabilities = np.arange(0.3, 0.61, 0.015)
#
range_synaptic_weights = np.arange(0.2, 0.4, 0.1)
range_number_neurons = np.arange(100, 200, 20)
range_connection_probabilities = np.arange(0.3, 0.61, 0.15)


def experiment_connection_probabilities():
    number_neurons = np.ones(range_connection_probabilities.size) * default_number_neurons
    synaptic_weights = np.ones(range_connection_probabilities.size) * default_synaptic_weights
    late_responses = calculate_late_responses_experiment(number_neurons,
                                                         range_connection_probabilities,
                                                         synaptic_weights)
    return range_connection_probabilities, late_responses


def plot(number_neurons, late_responses, xlabel, ylabel):
    plt.figure(figsize=(4, 3))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.plot(number_neurons, late_responses, '-')
    plt.plot(number_neurons, late_responses, 'o')
    plt.savefig('results/%s_by_%s.png' %
                (ylabel.lower().replace(' ', '_'),
                 xlabel.lower().replace(' ', '_')))


def exercise_2():
    # connection_probabilities, late_responses = experiment_connection_probabilities()
    # plot(connection_probabilities, late_responses, 'Connection Probability', 'Late Response (Hz)')

    tend = 3000
    f = plt.figure(figsize=(4, 3))

    for cp in [0.35, 0.37, 0.45, 0.50]:
        _, monitor = run_sim(default_number_neurons, cp, default_synaptic_weights, tend)
        plt.plot(monitor.times * 1000, monitor.rate, label='Connection Probability %d%%' % (cp * 100))

    plt.title('Population Rate')
    plt.ylabel('Rate (Hz)')
    plt.xlabel('Time (ms)')
    plt.xlim(xmin=0, xmax=tend)
    f.tight_layout()
    plt.legend(prop={'size': 8})
    plt.savefig('results/population_rate_compare_conn_prob.png')


def experiment_synaptic_weights():
    number_neurons = np.ones(range_synaptic_weights.size) * default_number_neurons
    connection_probabilities = np.ones(range_synaptic_weights.size) * default_connection_probabilities
    late_responses = calculate_late_responses_experiment(number_neurons,
                                                         connection_probabilities,
                                                         range_synaptic_weights)

    return range_synaptic_weights, late_responses


def exercise_3():
    synaptic_weights, late_responses = experiment_synaptic_weights()
    plot(synaptic_weights, late_responses, 'Synaptic weights (nA)', 'Late Response (Hz)')


def experiment_number_neurons():
    connection_probabilities = np.ones(range_number_neurons.size) * default_connection_probabilities
    synaptic_weights = np.ones(range_number_neurons.size) * default_synaptic_weights
    late_responses = calculate_late_responses_experiment(range_number_neurons,
                                                         connection_probabilities,
                                                         synaptic_weights)

    return range_number_neurons, late_responses


def exercise_4():
    number_neurons, late_responses = experiment_number_neurons()
    plot(number_neurons, late_responses, 'Number of Neurons', 'Late Response (Hz)')


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
    syn_current = np.arange(0, 30, 0.1)  # nano-amps

    for cp in [default_connection_probabilities, 0.365, 0.45]:
        effective_coefficient = compute_effective_coefficient(default_number_neurons,
                                                              cp,
                                                              default_synaptic_weights)
        activity = syn_current / effective_coefficient
        line = plt.plot(
            syn_current, activity,
            label='From Syn Current (Coefficient %f)' % effective_coefficient)

        intersections = get_intersections(lambda curr: curr / effective_coefficient,
                                          lambda curr: np.interp(curr, gain[0], gain[1]),
                                          gain[0].min(), gain[1].max())

        for i in range(intersections.shape[1]):
            print 'intersection:', intersections[0][i], intersections[1][i]

        plt.plot(intersections[0], intersections[1], 'o', color=line[0].get_color())


def exercise_5():
    plt.figure(figsize=(5, 3))
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

    # plt.figure(figsize=(4, 3))
    # max_coef = 0
    #
    # print 'experiment_connection_probabilities'
    # connection_probabilities, late_responses = experiment_connection_probabilities()
    # coefficient = compute_effective_coefficient(
    #     default_number_neurons, connection_probabilities, default_synaptic_weights)
    # plt.plot(coefficient, late_responses, label='Connection Probability')
    # max_coef = max(coefficient.max(), max_coef)
    #
    # print 'experiment_synaptic_weights'
    # synaptic_weights, late_responses = experiment_synaptic_weights()
    # coefficient = compute_effective_coefficient(
    #     default_number_neurons, default_connection_probabilities, synaptic_weights)
    # plt.plot(coefficient, late_responses, label='Synaptic Weights')
    # max_coef = max(coefficient.max(), max_coef)
    #
    # print 'experiment_number_neurons'
    # number_neurons, late_responses = experiment_number_neurons()
    # coefficient = compute_effective_coefficient(
    #     number_neurons, default_connection_probabilities, default_synaptic_weights)
    # plt.plot(coefficient, late_responses, label='Number of Neurons')
    # max_coef = max(coefficient.max(), max_coef)
    #
    # plt.xlabel('Effective Coefficient')
    # plt.ylabel('Late Response (Hz)')
    # plt.tight_layout()
    # plt.xlim(xmax=max_coef)
    # plt.legend(loc='lower right', prop={'size': 8})
    # plt.savefig('results/late_response_(hz)_by_effective_coefficient.png')

    external_current_strengths = np.arange(1.56, 1.6, 0.01)
    initial_response = np.zeros_like(external_current_strengths)
    late_response = np.zeros_like(external_current_strengths)

    f0 = plt.figure()#figsize=(4, 3))

    survive = []
    die = []
    for i, curr in enumerate(external_current_strengths):
        _, monitor = run_sim(default_number_neurons, 0.375, default_synaptic_weights, 300, curr)
        initial_response[i] = calculate_late_response(monitor, 0.075, 0.125)
        late_response[i] = calculate_late_response(monitor)

        if late_response[i] > 1:
            survive.append((curr, monitor))
        else:
            die.append((curr, monitor))

    for curr, monitor in survive:
        plt.plot(monitor.times * 1000, monitor.rate, 'g', label=str(curr))

    for curr, monitor in die:
        plt.plot(monitor.times * 1000, monitor.rate, 'r', label=str(curr))

    plt.title('Population Rate')
    plt.ylabel('Rate (Hz)')
    plt.xlabel('External Current (nA)')
    # plt.xlim(xmin=0, xmax=300)
    f0.tight_layout()
    plt.legend(loc='lower right', prop={'size': 8})
    plt.savefig('results/population_rate_by_current.png')

    #
    # f1 = plt.figure()#figsize=(4, 3))
    print initial_response
    print late_response
    # l = plt.plot(external_current_strengths, initial_response, '-', label='Initial Response')
    # plt.plot(external_current_strengths, initial_response, 'o', color=l[0].get_color())
    #
    # l = plt.plot(external_current_strengths, late_response, '-', label='Late Response')
    # plt.plot(external_current_strengths, late_response, 'o', color=l[0].get_color())
    #
    # plt.title('Population Rate')
    # plt.ylabel('Rate (Hz)')
    # plt.xlabel('External Current (nA)')
    # # plt.xlim(xmin=0, xmax=300)
    # f1.tight_layout()
    # plt.legend(loc='lower right', prop={'size': 8})
    # plt.savefig('results/late_response_(hz)_by_current.png')


def main():
    exercises = [
        # exercise_1,
        # exercise_2,
        # exercise_3,
        # exercise_4,
        exercise_5
    ]

    for i, excercise in enumerate(exercises):
        print 'excercise', i + 1
        excercise()

    print 'done'


main()

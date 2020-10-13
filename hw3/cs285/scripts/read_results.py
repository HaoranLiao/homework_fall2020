import glob
import tensorflow as tf

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    Z = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            # elif v.tag == 'Eval_AverageReturn':
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
            elif v.tag == 'Train_BestReturn':
                Z.append(v.simple_value)
    return X, Y, Z

if __name__ == '__main__':
    import glob
    from matplotlib import pyplot as plt
    import numpy as np

    logdir = '/Users/haoran/Documents/GitHub/homework_fall2020/hw3/cs285/data/q1_MsPacman-v0_03-10-2020_19-33-29/events*'
    logdir1 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw3/cs285/data/q2_dqn_1_LunarLander-v3_03-10-2020_14-10-21/events*'
    eventfile1 = glob.glob(logdir1)[0]
    logdir2 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw3/cs285/data/q2_dqn_2_LunarLander-v3_03-10-2020_15-40-11/events*'
    eventfile2 = glob.glob(logdir2)[0]
    logdir3 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw3/cs285/data/q2_dqn_3_LunarLander-v3_03-10-2020_17-21-44/events*'
    eventfile3 = glob.glob(logdir3)[0]
    logdir4 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw3/cs285/data/q2_doubledqn_1_LunarLander-v3_12-10-2020_20-17-24/events*'
    eventfile4 = glob.glob(logdir4)[0]
    logdir5 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw3/cs285/data/q2_doubledqn_2_LunarLander-v3_12-10-2020_21-09-00/events*'
    eventfile5 = glob.glob(logdir5)[0]
    logdir6 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw3/cs285/data/q2_doubledqn_3_LunarLander-v3_12-10-2020_21-54-12/events*'
    eventfile6 = glob.glob(logdir6)[0]

    X1, Y1, Z1 = get_section_results(eventfile1)
    X2, Y2, Z2 = get_section_results(eventfile2)
    X3, Y3, Z3 = get_section_results(eventfile3)
    high_van = [0] * len(Y1)
    low_van = [0] * len(Y1)
    avg_van = [0] * len(Y1)
    for i, (x, y) in enumerate(zip(X1, Y1)):
        #  print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
        high_van[i] = np.max([Y1[i], Y2[i], Y3[i]])
        low_van[i] = np.min([Y1[i], Y2[i], Y3[i]])
        avg_van[i] = np.mean([Y1[i], Y2[i], Y3[i]])

    X4, Y4, Z4 = get_section_results(eventfile4)
    X5, Y5, Z5 = get_section_results(eventfile5)
    X6, Y6, Z6 = get_section_results(eventfile6)
    high_ddq = [0] * len(Y4)
    low_ddq = [0] * len(Y4)
    avg_ddq = [0] * len(Y4)
    for i, (x, y) in enumerate(zip(X4, Y4)):
        arr = [Y4[i], Y5[i], Y6[i]]
        high_ddq[i] = np.max(arr)
        low_ddq[i] = np.min(arr)
        avg_ddq[i] = np.mean(arr)


    fig, ax = plt.subplots()
    plt.title('q2_dqn_LunarLandr-v3')
    ax.plot(X1[1:], avg_van, label='Vanilla DQN')
    ax.fill_between(X1[1:], low_van, high_van, alpha=0.2)
    ax.plot(X4[1:], avg_ddq, label='DDQN')
    ax.fill_between(X4[1:], low_ddq, high_ddq, alpha=0.2)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Average training rewards')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.show()
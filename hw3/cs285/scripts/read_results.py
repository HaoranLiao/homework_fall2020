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

    # logdir = '/Users/haoran/Documents/GitHub/homework_fall2020/hw3/cs285/data/q1_MsPacman-v0_03-10-2020_19-33-29/events*'
    logdir1 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw3/cs285/data/q2_dqn_1_LunarLander-v3_03-10-2020_14-10-21/events*'
    eventfile1 = glob.glob(logdir1)[0]
    logdir2 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw3/cs285/data/q3_hparam1_LunarLander-v3_12-10-2020_22-41-57/events*'
    eventfile2 = glob.glob(logdir2)[0]
    logdir3 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw3/cs285/data/q3_hparam2_LunarLander-v3_12-10-2020_22-59-50/events*'
    eventfile3 = glob.glob(logdir3)[0]
    logdir4 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw3/cs285/data/q3_hparam3_LunarLander-v3_12-10-2020_23-33-22/events*'
    eventfile4 = glob.glob(logdir4)[0]
    # logdir5 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw3/cs285/data/q2_doubledqn_2_LunarLander-v3_12-10-2020_21-09-00/events*'
    # eventfile5 = glob.glob(logdir5)[0]
    # logdir6 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw3/cs285/data/q2_doubledqn_3_LunarLander-v3_12-10-2020_21-54-12/events*'
    # eventfile6 = glob.glob(logdir6)[0]

    X1, Y1, Z1 = get_section_results(eventfile1)
    X2, Y2, Z2 = get_section_results(eventfile2)
    X3, Y3, Z3 = get_section_results(eventfile3)

    X4, Y4, Z4 = get_section_results(eventfile4)
    # X5, Y5, Z5 = get_section_results(eventfile5)
    # X6, Y6, Z6 = get_section_results(eventfile6)
    # high_ddq = [0] * len(Y4)
    # low_ddq = [0] * len(Y4)
    # avg_ddq = [0] * len(Y4)
    # for i, (x, y) in enumerate(zip(X4, Y4)):
    #     arr = [Y4[i], Y5[i], Y6[i]]
    #     high_ddq[i] = np.max(arr)
    #     low_ddq[i] = np.min(arr)
    #     avg_ddq[i] = np.mean(arr)


    fig, ax = plt.subplots()
    plt.title('q3_LunarLandr-v3_experimenting_w_hyperparameters')
    ax.plot(X1[1:], Y1, label='target update every 3000 (default)')
    # ax.fill_between(X1[1:], low_van, high_van, alpha=0.2)
    ax.plot(X2[1:], Y2, label='target update every 500')
    # ax.fill_between(X4[1:], low_ddq, high_ddq, alpha=0.2)
    ax.plot(X3[1:], Y3, label='target update every 10000')
    ax.plot(X4[1:], Y4, label='target update every 1500')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Average training rewards')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.show()
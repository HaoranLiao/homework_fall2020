import glob
import tensorflow as tf

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

if __name__ == '__main__':
    import glob
    from matplotlib import pyplot as plt

    # logdir1 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw4/data/hw4_q4_reacher_horizon5_reacher-cs285-v0_31-10-2020_14-54-14/events*'
    # eventfile1 = glob.glob(logdir1)[0]
    # logdir2 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw4/data/hw4_q4_reacher_horizon15_reacher-cs285-v0_31-10-2020_15-56-25/events*'
    # eventfile2 = glob.glob(logdir2)[0]
    # logdir3 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw4/data/hw4_q4_reacher_horizon30_reacher-cs285-v0_31-10-2020_17-54-17/events*'
    # eventfile3 = glob.glob(logdir3)[0]

    # logdir1 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw4/data/hw4_q4_reacher_numseq100_reacher-cs285-v0_31-10-2020_17-34-07/events*'
    # eventfile1 = glob.glob(logdir1)[0]
    # logdir2 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw4/data/hw4_q4_reacher_numseq1000_reacher-cs285-v0_31-10-2020_17-35-08/events*'
    # eventfile2 = glob.glob(logdir2)[0]

    logdir1 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw4/data/hw4_q4_reacher_ensemble1_reacher-cs285-v0_31-10-2020_17-37-09/events*'
    eventfile1 = glob.glob(logdir1)[0]
    logdir2 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw4/data/hw4_q4_reacher_ensemble3_reacher-cs285-v0_31-10-2020_17-38-52/events*'
    eventfile2 = glob.glob(logdir2)[0]
    logdir3 = '/Users/haoran/Documents/GitHub/homework_fall2020/hw4/data/hw4_q4_reacher_ensemble5_reacher-cs285-v0_31-10-2020_17-39-37/events*'
    eventfile3 = glob.glob(logdir3)[0]

    X1, Y1 = get_section_results(eventfile1)
    X2, Y2 = get_section_results(eventfile2)
    X3, Y3 = get_section_results(eventfile3)
    plt.figure()
    plt.title('Efffect of ensemble size')
    # for i, (x, y) in enumerate(zip(X1, Y1)):
        # print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
    plt.plot(X1, Y1, label='ensemble size 1')
    plt.plot(X2, Y2, label='ensemble size 3')
    plt.plot(X3, Y3, label='ensemble size 5')

    plt.xlabel('Iterations')
    plt.ylabel('Average evaluation return')
    plt.legend()
    plt.show()
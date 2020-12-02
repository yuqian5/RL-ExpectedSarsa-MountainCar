from agent import ExpectedSarsaAgent, MountainCarTileCoder
import MontainCarEnv
import matplotlib.pyplot as plt
import multiprocessing


def plot(data):
    plt.plot(data)
    plt.xlabel('episodes')
    plt.ylabel('steps')
    plt.show()


def plot_with_title(data, title):
    plt.plot(data)
    plt.xlabel('episodes')
    plt.ylabel('steps')
    plt.title(title)
    plt.show()


def run_episode(w):
    agent_info = {"initial_weights": w}

    # init agent
    agent = ExpectedSarsaAgent()
    agent.agent_init(agent_info)

    # init env
    env = MontainCarEnv.MountainCar()

    # init initial S, A and R
    state = env.state
    action = agent.agent_start(state)
    reward = None

    while True:
        state, reward, done = env.step(action)
        if done:
            steps = len(env.position_list)
            return steps, agent.w
        else:
            action = agent.agent_step(reward, state)


def run(episode_count, fdMutex, show_plot=False):
    steps_per_episode = []
    run_count = 0  # num of runs complete
    w = 0.0  # persistent w, init to 0

    while run_count < episode_count:
        steps, w = run_episode(w)
        steps_per_episode.append(steps)

        run_count += 1

    if show_plot:
        plot(steps_per_episode)

    fdMutex.acquire()
    fd = open("ExperimentResult.tsv", "a")
    for i in steps_per_episode:
        fd.write(str(i) + "\t")
    fd.write("\n")
    fd.close()
    fdMutex.release()

def q3(num_runs):
    run_result = []
    processes = []

    lock = multiprocessing.Lock()

    for i in range(num_runs):
        print("Process for Run %d created" % i)
        # run independent runs in parallel
        p = multiprocessing.Process(target=run, args=(200, lock, False))
        p.start()
        processes.append(p)

    for process in processes:
        process.join()

    print("Retrieving Experiment Results")
    fd = open("ExperimentResult.tsv", "r")
    while True:
        result = fd.readline()

        if result:
            run_result.append(result.rstrip().split("\t"))
        else:
            break

    print("Calculating run average")
    run_average = []
    for i in range(200):
        run_vertical_sum = 0
        for j in range(num_runs):
            run_vertical_sum += int(run_result[j][i])

        run_average.append(run_vertical_sum / num_runs)

    plot(run_average)


def main():
    q3(5)
    q3(5)


if __name__ == '__main__':
    main()

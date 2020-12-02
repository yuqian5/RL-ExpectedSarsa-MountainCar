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


def run_episode(info):
    agent_info = info

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


def run(fdMutex, episode_count, alpha=0.1, epsilon=0.0, num_tilings=8, num_tiles=8, show_plot=False):
    steps_per_episode = []
    run_count = 0  # num of runs complete

    agent_info = {"alpha": alpha, "epsilon": epsilon, "num_tilings": num_tilings, "num_tiles": num_tiles, "initial_weights": 0.0}

    while run_count < episode_count:
        steps, w = run_episode(agent_info)
        agent_info["initial_weights"] = w
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


def q4(episode_count, alpha=0.1, epsilon=0.0, num_tilings=8, num_tiles=8):
    run_result = []
    processes = []

    lock = multiprocessing.Lock()

    for i in range(50):
        print("Process for Run %d created" % i)
        # run independent runs in parallel
        p = multiprocessing.Process(target=run, args=(lock, episode_count, alpha, epsilon, num_tilings, num_tiles, False))
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
    fd.close()
    open("ExperimentResult.tsv", "w").close()

    print("Calculating run average")
    run_average = []
    for i in range(200):
        run_vertical_sum = 0
        for j in range(50):
            run_vertical_sum += int(run_result[j][i])

        run_average.append(run_vertical_sum / 50)

    print(run_result)
    title = "alpha={a},epsilon={b},num_tilings={c},num_tiles={d}".format(a=alpha, b=epsilon, c=num_tilings, d=num_tiles)
    plot_with_title(run_average, title)


def main():
    q4(200, 0.2, 0.0, 8, 8)


if __name__ == '__main__':
    main()

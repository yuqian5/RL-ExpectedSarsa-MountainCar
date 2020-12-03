from agent import ExpectedSarsaAgent, MountainCarTileCoder
import MontainCarEnv
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np


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


def q4(episode_count, build_plot, alpha=0.1, epsilon=0.0, num_tilings=8, num_tiles=8, num_runs=50):
    run_result = []
    processes = []

    lock = multiprocessing.Lock()

    for i in range(num_runs):
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
        for j in range(num_runs):
            run_vertical_sum += int(run_result[j][i])

        run_average.append(run_vertical_sum / num_runs)

    # print(run_result)
    if (build_plot):
        title = "alpha={a},epsilon={b},num_tilings={c},num_tiles={d}".format(a=alpha, b=epsilon, c=num_tilings, d=num_tiles)
        plot_with_title(run_average, title)
    return run_average

def find_better_parameters(times, alpha, epsilon, num_tilings, num_tiles, draft):
    myP = q4(200, False, alpha, 0, 8, 8, times)
    
    draft_mean = np.sum(draft)/len(draft)
    print("mean for draft parameters:", draft_mean)
    myP_mean = np.sum(myP)/len(myP)
    print("mean for new parameters:", myP_mean)

    draft_variance = sum([((x - draft_mean) ** 2) for x in draft]) / len(draft) 
    draft_sdv = np.sqrt(draft_variance)
    draft_se = draft_sdv / (np.sqrt(times))
    print("stander error for draft parameters:", draft_se)

    myP_variance = sum([((x - myP_mean) ** 2) for x in myP]) / len(myP) 
    myP_sdv = np.sqrt(myP_variance)
    myP_se = myP_sdv / (np.sqrt(times))
    print("stander error for new parameters:", myP_se)
    
    print("start comparing:")
    mean_diff = draft_mean - myP_mean
    if mean_diff < 0:
        mean_diff = -mean_diff
    if (mean_diff > 2.5*(np.maximum(draft_se, myP_se))):
        print("Better")
        if_plot = True
    else:
        print("Not")
        if_plot = False
    
    return myP, if_plot

def main():
    #q4(200, 0.2, 0.0, 8, 8, 50)
    times = 50
    import q3
    draft = q3.q3(times, False)
    
    alpha = 0.1005
    epsilon = 0.0
    num_tilings = 8
    num_tiles = 8
    better_list, build_polt = find_better_parameters(times, alpha, epsilon, num_tilings, num_tiles, draft)
    if build_polt:
        title = "alpha={a},epsilon={b},num_tilings={c},num_tiles={d}".format(a=alpha, b=epsilon, c=num_tilings, d=num_tiles)
        plot_with_title(better_list, title)

if __name__ == '__main__':
    main()

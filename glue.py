import agent
import MontainCarEnv
import matplotlib.pyplot as plt


class glue:
    def __init__(self):
        self.agent = None
        self.env = MontainCarEnv.MountainCar()

        self.w = None

        self.step_per_episode = []

    def q3(self):
        run = []
        for i in range(50):
            print("***** RUN %d *****" % i)
            self.run(200)
            run.append(self.step_per_episode)
            self.w = None  # clear weight

        run_average = []
        for i in range(200):
            sum = 0
            for j in range(50):
                sum += run[j][i]

            run_average.append(sum / 50)

        self.plot(run_average)

    def q4(self, episode_count, alpha=0.1, epsilon=0.0, num_tilings=8, num_tiles=8):
        title = "alpha={a},epsilon={b},num_tilings={c},num_tiles={d}".format(a=alpha, b=epsilon, c=num_tilings, d=num_tiles)

        run = []
        for i in range(10):
            print("***** RUN %d *****" % i)
            self.q4_runner(episode_count, alpha, epsilon, num_tilings, num_tiles)
            run.append(self.step_per_episode)
            self.w = None  # clear weight

        run_average = []
        for i in range(200):
            sum = 0
            for j in range(10):
                sum += run[j][i]

            run_average.append(sum / 10)

        self.plot_with_title(run_average, title)

    def q4_runner(self, episode_count, alpha=0.1, epsilon=0.0, num_tilings=8, num_tiles=8, show_plot=False):
        run_count = 0

        agent_info = {"alpha": alpha, "epsilon": epsilon, "num_tilings": num_tilings, "num_tiles": num_tiles}
        print("Running episode %d" % run_count)
        self.agent = agent.ExpectedSarsaAgent()
        self.agent.agent_init(agent_info)

        self.run_episode(run_count)

        run_count += 1

        while run_count < episode_count:
            print("Running episode %d" % run_count)
            agent_info["initial_weights"] = self.w
            self.agent = agent.ExpectedSarsaAgent()
            self.agent.agent_init(agent_info)

            self.run_episode(run_count)

            run_count += 1

        if show_plot:
            self.plot(self.step_per_episode)

    def run(self, episode_count, show_plot=False):
        run_count = 0

        print("Running episode %d" % run_count)
        self.agent = agent.ExpectedSarsaAgent()
        self.agent.agent_init()

        self.run_episode(run_count)

        run_count += 1

        while run_count < episode_count:
            print("Running episode %d" % run_count)
            agent_info = {"initial_weights": self.w}
            self.agent = agent.ExpectedSarsaAgent()
            self.agent.agent_init(agent_info)

            self.run_episode(run_count)

            run_count += 1

        if show_plot:
            self.plot(self.step_per_episode)
        
    def run_episode(self, episode_num):
        state = self.env.state
        action = None
        reward = None

        action = self.agent.agent_start(state)
        while True:
            state, reward, done = self.env.step(action)
            if done:
                self.w = self.agent.w
                # self.env.render(file_path="./mountainCar" + str(episode_num) + ".gif", mode='gif')
                steps = len(self.env.position_list)
                # print("%d steps taken" % steps)
                self.step_per_episode.append(steps)
                self.env.reset()
                return 0
            else:
                action = self.agent.agent_step(reward, state)

    def plot(self, data):
        plt.plot(data)
        plt.xlabel('episodes')
        plt.ylabel('steps')
        plt.show()

    def plot_with_title(self, data, title):
        plt.plot(data)
        plt.xlabel('episodes')
        plt.ylabel('steps')
        plt.title(title)
        plt.show()


def main():
    g = glue()
    # g.q3()
    g.q4(200, 0.2, 0.0, 8, 8)


if __name__ == '__main__':
    main()

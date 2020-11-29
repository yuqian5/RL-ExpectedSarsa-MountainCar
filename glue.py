import agent
import MontainCarEnv

class glue:
    def __init__(self):
        self.agent = None
        self.env = MontainCarEnv.MountainCar()

        self.w = None

    def run(self, episode_count):
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

    def run_episode(self, episode_num):
        state = self.env.state
        action = None
        reward = None

        action = self.agent.agent_start(state)
        while True:
            state, reward, done = self.env.step(action)
            if done:
                self.w = self.agent.w
                self.env.render(file_path="./mountainCar" + str(episode_num) + ".gif", mode='gif')
                self.env.reset()
                return 0
            else:
                action = self.agent.agent_step(reward, state)


def main():
    g = glue()
    g.run(2)


if __name__ == '__main__':
    main()


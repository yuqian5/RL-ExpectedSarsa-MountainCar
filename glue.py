import agent
# TODO
# import env

class glue:
    def __init__(self):
        self.agent = agent.ExpectedSarsaAgent()
        self.agent.agent_init()

    '''
    The gist is...
    glue will start and create an instance of the agent and env
    
    then:
    
    glue call env to get start state
    glue call agent start with start state and get action
    
    loop:
    glue use action to call env, get reward and new state (position, velocity)
    glue uses (position, velocity) to call agent_step to get new action
    '''

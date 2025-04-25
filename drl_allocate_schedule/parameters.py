import job_distribution

class Parameters:
    def __init__(self):

        self.output_filename = 'model/tmp'

        self.num_epochs = 10000         # number of training epochs
        self.job_min_simu_len = 80
        self.job_max_simu_len = 100
        self.task_max_simu_len = 6
        self.task_min_simu_len = 1
        self.num_ex = 10                # number of sequences

        self.output_freq = 10          # interval for output and store parameters

        self.num_seq_per_batch = 10    # number of sequences to compute baseline
        self.episode_max_length = 200  # enforcing an artificial terminal
        self.job_num_cap = 50  # maximum number of distinct colors in current work graph

        self.num_res = 2               # number of resources in the system
        self.num_nw = 5                # maximum allowed number of work in the queue

        self.time_horizon = 20         # number of time steps in the graph
        self.max_task_len = 15          # maximum duration of new jobs
        self.max_task_size = 15         # maximum resource request of new work

        self.backlog_size = 80         # backlog queue size

        self.max_track_since_new = 10  # track how many time steps since last new jobs

        self.new_job_rate = 0.7        # lambda in new job arrival Poisson Process

        self.discount = 1           # discount factor

        # distribution for new job arrival
        self.dist = job_distribution.Dist(self.num_res, self.max_task_size, self.max_task_len)

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw + 1  # + 1 for void action

        self.delay_penalty = -1.1       # penalty for delaying things in the current work screen
        self.hold_penalty = -1.2        # penalty for holding things in the new work screen
        self.dismiss_penalty = -1.3     # penalty for missing a job because the queue is full

        self.num_frames = 1           # number of frames to combine and process
        self.lr_rate = 0.001          # learning rate
        self.rms_rho = 0.9            # for rms prop
        self.rms_eps = 1e-9           # for rms prop

        self.unseen = False  # change random seed to generate unseen example

        # supervised learning mimic policy
        self.batch_size = 10
        self.evaluate_policy_name = "SJF"

    def compute_dependent_parameters(self):

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw + 1  # + 1 for void action


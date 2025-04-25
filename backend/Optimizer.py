import torch
import os
import time
import threading

from environment import Env
import parameters
import pg_network

prefix = os.path.dirname(os.path.abspath(__file__))

class Optimizer:
    def __init__(self):
        self.pa = parameters.Parameters()
        self.pa.num_nw = 10
        self.pa.network_output_dim = self.pa.num_nw + 1
        self.pa.new_job_rate = 0.3
        self.pa.compute_dependent_parameters()

        self.env = Env(self.pa, job_seq=None, render=False, repre='image', end='no_new_task')

        actor_lr = 1e-3
        critic_lr = 3e-3
        allocate_lr = 1e-3
        hidden_dim = 256
        gamma = 0.98
        # for PPO
        lmbda = 0.95
        epochs = 10
        eps = 0.2
        allocate_eps = 0.2
        action_dim = 11
        allocate_action_dim = 17
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.schedule_actor = pg_network.GRPOForSchedule(hidden_dim, action_dim, actor_lr,
                                                    gamma, epochs, eps, device)
        self.allocate_actor = pg_network.GPROForAllocate(hidden_dim, allocate_action_dim, allocate_lr,
                                                    gamma, epochs, allocate_eps, device)
        self.schedule_actor.load_data(prefix + '/model/tmp_300_schedule_actor.pth')
        self.allocate_actor.load_data(prefix + '/model/tmp_300_allocate_actor.pth')
        self.schedule_actor.actor.eval()
        self.allocate_actor.actor.eval()

        # 启动调度线程
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()

    def _scheduler_loop(self):
        state = self.env.observe()
        for i in range(2000):
            action = self.schedule_actor.take_action(state)
            next_state, reward, done, info = self.env.schedule_step(action)

            if done:
                print("_scheduler_loop finished!")
                break

            state = next_state

            if (i+1) % 5 == 0:
                allocate_next_action = self.allocate_actor.take_action(state)
                _ = self.env.allocate_step(allocate_next_action)
                state = self.env.observe()

            time.sleep(0.1)

        print("_scheduler_loop finished!")





import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import environment
import job_distribution
import slow_down_cdf
import torch
import pg_network

def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert np.array(x).ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def get_traj(schedule_actor, allocate_actor, env, episode_max_length):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """
    env.reset()

    info = []
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    allocate_transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'costs': [], 'dones': []}

    state = env.observe()
    allocate_action = 6
    allocate_state = state
    schedule_reward = []

    for i in range(episode_max_length):

        action = schedule_actor.take_action(state)

        next_state, reward, done, info = env.schedule_step(action)
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)

        schedule_reward.append(reward)
        state = next_state

        if (i+1) % 5 == 0:
            allocate_next_action = allocate_actor.take_action(state)
            cost = env.allocate_step(allocate_next_action)

            allocate_transition_dict['states'].append(allocate_state)
            allocate_transition_dict['actions'].append(allocate_action)
            allocate_transition_dict['next_states'].append(state)
            allocate_transition_dict['rewards'].append(sum(schedule_reward) - 0.1*cost)
            allocate_transition_dict['costs'].append(cost)
            allocate_transition_dict['dones'].append(done)

            allocate_state = state
            allocate_action = allocate_next_action
            schedule_reward = []
            
            state = env.observe()

        if done:
            break

    return {'transition': transition_dict,
            'allocate_transition': allocate_transition_dict,
            'info': info}

def process_all_info(trajs):
    enter_time = []
    finish_time = []
    job_len = []

    for traj in trajs:
        enter_time.append(np.array([traj['info'].record[i].enter_time for i in range(len(traj['info'].record))]))
        finish_time.append(np.array([traj['info'].record[i].finish_time for i in range(len(traj['info'].record))]))
        job_len.append(np.array([traj['info'].record[i].len for i in range(len(traj['info'].record))]))

    enter_time = np.concatenate(enter_time)
    finish_time = np.concatenate(finish_time)
    job_len = np.concatenate(job_len)

    return enter_time, finish_time, job_len


def plot_lr_curve(output_file_prefix, max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                  max_rew_lr_curve_, mean_rew_lr_curve_, cost_lr_curve):
    num_colors = 2
    cm = plt.get_cmap('gist_rainbow')

    fig = plt.figure(figsize=(12, 9))

    ax = fig.add_subplot(221)
    ax.set_prop_cycle('color', [matplotlib.cm.viridis(1. * i / num_colors) for i in range(num_colors)])

    ax.plot(mean_rew_lr_curve, linewidth=2, label='PG mean')
    # for k in ref_discount_rews:
    #     ax.plot(np.tile(np.average(ref_discount_rews[k]), len(mean_rew_lr_curve)), linewidth=2, label=k)
    ax.plot(max_rew_lr_curve, linewidth=2, label='PG max')

    plt.legend(loc=4)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Discounted Total Reward", fontsize=20)

    ax = fig.add_subplot(222)
    ax.set_prop_cycle('color', [matplotlib.cm.viridis(1. * i / num_colors) for i in range(num_colors)])
    ax.plot(slow_down_lr_curve, linewidth=2, label='PG mean')
    # for k in ref_discount_rews:
    #     ax.plot(np.tile(np.average(np.concatenate(ref_slow_down[k])), len(slow_down_lr_curve)), linewidth=2, label=k)
    plt.legend(loc=1)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Slowdown", fontsize=20)
    
    ax = fig.add_subplot(223)
    ax.set_prop_cycle('color', [matplotlib.cm.viridis(1. * i / num_colors) for i in range(num_colors)])
    ax.plot(mean_rew_lr_curve_, linewidth=2, label='PG mean')
    ax.plot(max_rew_lr_curve_, linewidth=2, label='PG max')
    
    plt.legend(loc=4)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Discounted Total Reward", fontsize=20)
    
    ax = fig.add_subplot(224)
    ax.set_prop_cycle('color', [matplotlib.cm.viridis(1. * i / num_colors) for i in range(num_colors)])
    ax.plot(cost_lr_curve, linewidth=2, label='Cost mean')
    plt.legend(loc=4)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Cost", fontsize=20)



    plt.savefig(output_file_prefix + "_lr_curve" + ".pdf")


def launch(pa, pg_resume=None, render=False, repre='compact', end='no_new_job'):

    envs = []

    job_seqs = job_distribution.generate_sequence_work(pa)

    actor_lr = 1e-3
    critic_lr = 3e-3
    allocate_lr = 1e-3
    hidden_dim = 256
    gamma = 0.98
    # for PPO
    lmbda = 0.95
    epochs = 5
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for ex in range(pa.num_ex):

        print("-prepare for env-", ex)

        env = environment.Env(pa=pa, job_seq=job_seqs[ex],
                              render=False, repre=repre, end=end)
        envs.append(env)

    action_dim = pa.network_output_dim
    allocate_action_dim = 17

    schedule_agent = pg_network.PPO(hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)
    schedule_actor = pg_network.GRPOForSchedule(hidden_dim, action_dim, actor_lr, gamma, epochs, eps, device)
    allocate_eps = 0.2
    allocate_agent = pg_network.PPO(hidden_dim, allocate_action_dim, actor_lr, critic_lr, lmbda,
                epochs, allocate_eps, gamma, device)
    allocate_actor = pg_network.GPROForAllocate(hidden_dim, allocate_action_dim, allocate_lr, gamma, epochs, allocate_eps, device)


    if pg_resume is not None:
        pass

    mean_rew_lr_curve = []
    max_rew_lr_curve = []
    slow_down_lr_curve = []
    mean_rew_lr_curve_ = []
    max_rew_lr_curve_ = []
    cost_lr_curve = []

    # --------------------------------------
    print("Start training...")
    # --------------------------------------

    timer_start = time.time()

    for iteration in range(1, pa.num_epochs):

        ex_indices = list(range(pa.num_ex))
        np.random.shuffle(ex_indices)

        eprewlist = []
        eprewlist_ = []
        epcostlist = []
        eplenlist =[]
        slowdownlist =[]
        rew_adv_list = []
        rew_adv_list_ = []

        epoch = 10
        trajs = []
        for ex in range(pa.num_ex):
            ex_idx = ex_indices[ex]
            trajs_epoch = []
            for i in range(epoch):
                traj = get_traj(schedule_actor, allocate_actor, envs[ex_idx], pa.episode_max_length)
                trajs_epoch.append(traj)
                trajs.append(traj)

            rew_rets = [discount(np.array(traj['allocate_transition']['rewards']), 1) for traj in trajs_epoch]
            maxlen = max(len(ret) for ret in rew_rets)
            padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rew_rets]
            baseline = np.mean(padded_rets, axis=0)
            # std = np.std(padded_rets, axis=0)
            # std += 1e-5
            rew_advs = [ret - baseline[:len(ret)] for ret in rew_rets]
            rew_adv_list.append(np.concatenate(rew_advs))
            
            # rew_rets = np.concatenate([traj['allocate_transition']['rewards'] for traj in trajs_epoch])
            # baseline = np.mean(rew_rets, axis=0)
            # std = np.std(rew_rets, axis=0)
            # std += 1e-5
            # rew_advs = [(ret - baseline) / std for ret in rew_rets]
            # rew_adv_list.append(rew_advs)

            rew_rets_ = [discount(np.array(traj['transition']['rewards']), 1) for traj in trajs_epoch]
            maxlen_ = max(len(ret) for ret in rew_rets_)
            padded_rets_ = [np.concatenate([ret, np.zeros(maxlen_ - len(ret))]) for ret in rew_rets_]
            baseline_ = np.mean(padded_rets_, axis=0)
            # std = np.std(padded_rets_, axis=0)
            # std += 1e-5
            rew_advs_ = [ret - baseline_[:len(ret)] for ret in rew_rets_]
            rew_adv_list_.append(np.concatenate(rew_advs_))

            all_eprews = np.array([discount(traj['transition']['rewards'], pa.discount)[0] for traj in trajs_epoch])
            all_eprews_ = np.array([discount(traj['allocate_transition']['rewards'], pa.discount)[0] for traj in trajs_epoch])
            all_epcosts = np.array([np.sum(traj['allocate_transition']['costs']) for traj in trajs_epoch])
            all_eplens = np.array([len(traj['transition']['rewards']) for traj in trajs_epoch])  # episode lengths
            # All Job Stat
            enter_time, finish_time, job_len = process_all_info(trajs_epoch)
            finished_idx = (finish_time >= 0)
            all_slowdown = (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]

            eprewlist.append(all_eprews)
            eprewlist_.append(all_eprews_)
            epcostlist.append(all_epcosts)
            eplenlist.append(all_eplens)
            slowdownlist.append(all_slowdown)

        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

        max_y = max(state.shape[-1] for traj in trajs for state in traj['transition']['states'])
        states_padded = [np.pad(state, ((0, 0), (0, 0), (0, max_y - state.shape[-1])), mode='constant', constant_values=-1)
                         for traj in trajs for state in traj['transition']['states']]
        max_y_ = max(state.shape[-1] for traj in trajs for state in traj['transition']['next_states'])
        states_padded_ = [np.pad(state, ((0, 0), (0, 0), (0, max_y_ - state.shape[-1])), mode='constant', constant_values=-1)
                          for traj in trajs for state in traj['transition']['next_states']]

        # transition_dict['states'] = np.array(states_padded)
        transition_dict['states'] = np.concatenate([traj['transition']['states'] for traj in trajs])
        transition_dict['actions'] = np.concatenate([traj['transition']['actions'] for traj in trajs], dtype=np.int64)
        # transition_dict['next_states'] = np.array(states_padded_)
        transition_dict['next_states'] = np.concatenate([traj['transition']['next_states'] for traj in trajs])
        transition_dict['rewards'] = np.concatenate(rew_adv_list_)
        # transition_dict['rewards'] = np.concatenate([traj['transition']['rewards'] for traj in trajs])
        transition_dict['dones'] = np.concatenate([traj['transition']['dones'] for traj in trajs])


        allocate_transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'costs': [], 'dones': []}

        max_y = max(state.shape[-1] for traj in trajs for state in traj['allocate_transition']['states'])
        states_padded = [np.pad(state, ((0, 0), (0, 0), (0, max_y - state.shape[-1])), mode='constant', constant_values=-1)
                         for traj in trajs for state in traj['allocate_transition']['states']]
        max_y_ = max(state.shape[-1] for traj in trajs for state in traj['allocate_transition']['next_states'])
        states_padded_ = [np.pad(state, ((0, 0), (0, 0), (0, max_y_ - state.shape[-1])), mode='constant', constant_values=-1)
                          for traj in trajs for state in traj['allocate_transition']['next_states']]

        # allocate_transition_dict['states'] = np.array(states_padded)
        allocate_transition_dict['states'] = np.concatenate([traj['allocate_transition']['states'] for traj in trajs])
        allocate_transition_dict['actions'] = np.concatenate([traj['allocate_transition']['actions'] for traj in trajs], dtype=np.int64)
        # allocate_transition_dict['next_states'] = np.array(states_padded_)
        allocate_transition_dict['next_states'] = np.concatenate([traj['allocate_transition']['next_states'] for traj in trajs])
        allocate_transition_dict['rewards'] = np.concatenate(rew_adv_list)
        # allocate_transition_dict['rewards'] = np.concatenate([traj['allocate_transition']['rewards'] for traj in trajs])
        allocate_transition_dict['dones'] = np.concatenate([traj['allocate_transition']['dones'] for traj in trajs])
        
        schedule_actor.update(transition_dict)
        allocate_actor.update(allocate_transition_dict)
        # schedule_agent.update(transition_dict)
        # allocate_agent.update(allocate_transition_dict)
        

        timer_end = time.time()

        print("-----------------")
        print("Iteration: \t %i" % iteration)
        print("NumTimesteps: \t %i" % np.sum(eplenlist))
        print("MaxRew: \t %s" % np.average([np.max(rew) for rew in eprewlist]))
        print("MeanRew: \t %s +- %s" % (np.mean(eprewlist), np.std(eprewlist)))
        print("MeanSlowdown: \t %s" % np.mean([np.mean(sd) for sd in slowdownlist]))
        print("MeanCost: \t %s +- %s" % (np.mean(epcostlist), np.std(epcostlist)))
        print("MeanLen: \t %s +- %s" % (np.mean(eplenlist), np.std(eplenlist)))
        print("Elapsed time: \t %s" % (timer_end - timer_start), "seconds")
        print("-----------------")

        timer_start = time.time()

        max_rew_lr_curve.append(np.average([np.max(rew) for rew in eprewlist]))
        mean_rew_lr_curve.append(np.mean(eprewlist))
        slow_down_lr_curve.append(np.mean([np.mean(sd) for sd in slowdownlist]))
        max_rew_lr_curve_.append(np.average([np.max(rew) for rew in eprewlist_]))
        mean_rew_lr_curve_.append(np.mean(eprewlist_))
        cost_lr_curve.append(np.mean(epcostlist))

        if iteration % pa.output_freq == 0:

            schedule_actor.save_data(pa.output_filename + '_' + str(iteration))
            allocate_actor.save_data(pa.output_filename + '_' + str(iteration))

            pa.unseen = True
            # slow_down_cdf.launch(pa, pa.output_filename + '_' + str(iteration) + '.ckpt',
                                # render=False, plot=True, repre=repre, end=end)
            pa.unseen = False
            # test on unseen examples

            plot_lr_curve(pa.output_filename,
                          max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                          max_rew_lr_curve_, mean_rew_lr_curve_, cost_lr_curve)


def main():

    import parameters

    pa = parameters.Parameters()

    pa.num_ex = 10  # 100
    pa.num_nw = 10
    pa.network_output_dim = pa.num_nw + 1
    pa.num_seq_per_batch = 10
    pa.output_freq = 50
    pa.batch_size = 10
    
    # pa.max_nw_size = 5
    # pa.job_len = 5
    pa.new_job_rate = 0.3

    pa.episode_max_length = 2000  # 2000

    pa.compute_dependent_parameters()

    pg_resume = None
    # pg_resume = 'data/tmp_450.pkl'

    render = False

    launch(pa, pg_resume, render, repre='image', end='no_new_job')


if __name__ == '__main__':
    main()

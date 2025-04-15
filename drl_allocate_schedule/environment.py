import numpy as np
import math

class Env:
    def __init__(self, pa, job_seq = None,
                 seed=42, render=False, repre='compact', end='no_new_task'):

        self.pa = pa
        self.job_seq = job_seq
        self.nw_len_seqs = np.empty((0,), dtype=int)
        self.nw_size_seqs = np.empty((0,2), dtype=int)
        self.render = render
        self.repre = repre  # image or compact representation
        self.end = end  # termination type, 'no_new_task' or 'all_done'

        self.nw_dist = pa.dist.bi_model_dist

        self.curr_time = 0

        # set up random seed
        if self.pa.unseen:
            np.random.seed(314159)
        else:
            np.random.seed(seed)

        self.seq_idx = 0  # index in that sequence
        self.simu_job_len = len(job_seq)


        # initialize system
        self.machine_cluster = MachineCluster(pa)
        self.task_slot = TaskSlot(pa)
        self.task_backlog = TaskBacklog(pa)
        self.task_record = TaskRecord()
        self.extra_info = ExtraInfo(pa)

        self.generate_task_seq()

    def generate_task_seq(self):
        for job in self.job_seq:
            if job is None:
                self.nw_len_seqs = np.append(self.nw_len_seqs, 0)
                self.nw_size_seqs = np.append(self.nw_size_seqs, [[0, 0]], axis=0)
                continue
            self.nw_len_seqs = np.append(self.nw_len_seqs, job[0])
            self.nw_size_seqs = np.append(self.nw_size_seqs, job[1], axis=0)

    def get_new_task_from_seq(self, seq_idx):
        new_task = Task(res_vec=self.nw_size_seqs[seq_idx, :],
                      task_len=self.nw_len_seqs[seq_idx],
                      task_id=len(self.task_record.record),
                      enter_time=self.curr_time)
        return new_task

    def observe(self):
        if self.repre == 'image':
            backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))

            network_input_width = int((self.machine_cluster.res_slot + self.pa.max_task_size * self.pa.num_nw)
                                       * self.pa.num_res + self.pa.backlog_size/self.pa.time_horizon + 3)
            image_repr = np.zeros((self.pa.time_horizon, network_input_width))

            ir_pt = 0

            for i in range(self.pa.num_res):

                image_repr[:, ir_pt: ir_pt + self.machine_cluster.res_slot] = self.machine_cluster.canvas[i, :, :]
                ir_pt += self.machine_cluster.res_slot

                for j in range(self.pa.num_nw):

                    if self.task_slot.slot[j] is not None:  # fill in a block of work
                        image_repr[: self.task_slot.slot[j].len, ir_pt: ir_pt + self.task_slot.slot[j].res_vec[i]] = 1

                    ir_pt += self.pa.max_task_size

            image_repr[: int(self.task_backlog.curr_size / backlog_width),
                       ir_pt: ir_pt + backlog_width] = 1

            if self.task_backlog.curr_size % backlog_width > 0:
                image_repr[int(self.task_backlog.curr_size / backlog_width),
                ir_pt: ir_pt + self.task_backlog.curr_size % backlog_width] = 1
            ir_pt += backlog_width

            image_repr[:, ir_pt: ir_pt + 1] = self.extra_info.time_since_last_new_task / \
                                              float(self.extra_info.max_tracking_time_since_last_task)

            ir_pt += 1
            mtype_cnt = [0, 0]
            for m in self.machine_cluster.machine_cluster:
                mtype_cnt[m.type] += 1
            image_repr[:, ir_pt: ir_pt + 1] = mtype_cnt[0] / 5
            ir_pt += 1
            image_repr[:, ir_pt: ir_pt + 1] = mtype_cnt[0] / 2

            ir_pt += 1

            assert ir_pt == image_repr.shape[1]

            return np.expand_dims(image_repr, axis=0)

        elif self.repre == 'compact':
            compact_repr = np.zeros(self.pa.time_horizon * (self.pa.num_res + 1) +  # current work
                                    self.pa.num_nw * (self.pa.num_res + 1) +  # new work
                                    1,  # backlog indicator
                                    dtype=float)

            cr_pt = 0

            # current work reward, after each time step, how many tasks left in the machine
            task_allocated = np.ones(self.pa.time_horizon) * len(self.machine_cluster.running_task)
            for j in self.machine_cluster.running_task:
                task_allocated[j.finish_time - self.curr_time:] -= 1

            compact_repr[cr_pt: cr_pt + self.pa.time_horizon] = task_allocated
            cr_pt += self.pa.time_horizon

            # current work available slots
            for i in range(self.pa.num_res):
                compact_repr[cr_pt: cr_pt + self.pa.time_horizon] = self.machine_cluster.avbl_slot[:, i]
                cr_pt += self.pa.time_horizon

            # new work duration and size
            for i in range(self.pa.num_nw):

                if self.task_slot.slot[i] is None:
                    compact_repr[cr_pt: cr_pt + self.pa.num_res + 1] = 0
                    cr_pt += self.pa.num_res + 1
                else:
                    compact_repr[cr_pt] = self.task_slot.slot[i].len
                    cr_pt += 1

                    for j in range(self.pa.num_res):
                        compact_repr[cr_pt] = self.task_slot.slot[i].res_vec[j]
                        cr_pt += 1

            # backlog queue
            compact_repr[cr_pt] = self.task_backlog.curr_size
            cr_pt += 1

            assert cr_pt == len(compact_repr)  # fill up the compact representation vector

            return compact_repr

    def get_reward(self):

        reward = 0
        for j in self.machine_cluster.running_task:
            reward += self.pa.delay_penalty / float(j.len)

        for j in self.task_slot.slot:
            if j is not None:
                reward += self.pa.hold_penalty / float(j.len)

        for j in self.task_backlog.backlog:
            if j is not None:
                reward += self.pa.dismiss_penalty / float(j.len)

        return reward

    def schedule_step(self, a):

        status = None

        done = False
        reward = 0
        info = None

        if a == self.pa.num_nw:  # explicit void action
            status = 'MoveOn'
        elif self.task_slot.slot[a] is None:  # implicit void action
            status = 'MoveOn'
        else:
            allocated = self.machine_cluster.allocate_task(self.task_slot.slot[a], self.curr_time)
            if not allocated:  # implicit void action
                status = 'MoveOn'
            else:
                status = 'Allocate'

        if status == 'MoveOn':
            self.curr_time += 1
            self.machine_cluster.time_proceed(self.curr_time)
            self.extra_info.time_proceed()

            # add new Tasks
            if self.seq_idx < len(self.nw_len_seqs) - 1:
                self.seq_idx += 1
            else:
                done = True

            if not done:
                if self.seq_idx < len(self.nw_len_seqs):  # otherwise, end of new Task sequence, i.e. no new Tasks
                    new_task = self.get_new_task_from_seq(self.seq_idx)

                    if new_task.len > 0:  # a new task comes
                        to_backlog = True
                        for i in range(self.pa.num_nw):
                            if self.task_slot.slot[i] is None:  # put in new visible Task slots
                                self.task_slot.slot[i] = new_task
                                self.task_record.record[new_task.id] = new_task
                                to_backlog = False
                                break

                        if to_backlog:
                            if self.task_backlog.curr_size < self.pa.backlog_size:
                                self.task_backlog.backlog[self.task_backlog.curr_size] = new_task
                                self.task_backlog.curr_size += 1
                                self.task_record.record[new_task.id] = new_task
                            else:  # abort, backlog full
                                print("Backlog is full.")
                                # exit(1)

                        self.extra_info.new_task_comes()

            reward = self.get_reward()

        elif status == 'Allocate':
            self.task_record.record[self.task_slot.slot[a].id] = self.task_slot.slot[a]
            self.task_slot.slot[a] = None

            # dequeue backlog
            if self.task_backlog.curr_size > 0:
                self.task_slot.slot[a] = self.task_backlog.backlog[0]  # if backlog empty, it will be 0
                self.task_backlog.backlog[: -1] = self.task_backlog.backlog[1:]
                self.task_backlog.backlog[-1] = None
                self.task_backlog.curr_size -= 1

        ob = self.observe()

        info = self.task_record

        if done:
            self.seq_idx = 0

            self.reset()

        return ob, reward, done, info

    def allocate_step(self, a):
        cost = 0
        mtype_cnt = [0, 0]
        for m in self.machine_cluster.machine_cluster:
            td = self.curr_time - m.last_account_time
            cost += td * m.price_per_time
            m.last_account_time = self.curr_time

            mtype_cnt[m.type] += 1

        if a == 0:
            self.machine_cluster.allocate_resource(0, 2 - mtype_cnt[0], self.curr_time)
            self.machine_cluster.allocate_resource(1, 1 - mtype_cnt[1], self.curr_time)
        elif a == 1:
            self.machine_cluster.allocate_resource(0, 3 - mtype_cnt[0], self.curr_time)
            self.machine_cluster.allocate_resource(1, 1 - mtype_cnt[1], self.curr_time)
        elif a == 2:
            self.machine_cluster.allocate_resource(0, 4 - mtype_cnt[0], self.curr_time)
            self.machine_cluster.allocate_resource(1, 1 - mtype_cnt[1], self.curr_time)
        elif a == 3:
            self.machine_cluster.allocate_resource(0, 5 - mtype_cnt[0], self.curr_time)
            self.machine_cluster.allocate_resource(1, 1 - mtype_cnt[1], self.curr_time)
        elif a == 4:
            self.machine_cluster.allocate_resource(0, 2 - mtype_cnt[0], self.curr_time)
            self.machine_cluster.allocate_resource(1, 2 - mtype_cnt[1], self.curr_time)
        elif a == 5:
            self.machine_cluster.allocate_resource(0, 3 - mtype_cnt[0], self.curr_time)
            self.machine_cluster.allocate_resource(1, 2 - mtype_cnt[1], self.curr_time)
        elif a == 6:
            self.machine_cluster.allocate_resource(0, 4 - mtype_cnt[0], self.curr_time)
            self.machine_cluster.allocate_resource(1, 2 - mtype_cnt[1], self.curr_time)
        elif a == 7:
            self.machine_cluster.allocate_resource(0, 5 - mtype_cnt[0], self.curr_time)
            self.machine_cluster.allocate_resource(1, 2 - mtype_cnt[1], self.curr_time)

        return cost

    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0

        # initialize system
        self.machine_cluster = MachineCluster(self.pa)
        self.task_slot = TaskSlot(self.pa)
        self.task_backlog = TaskBacklog(self.pa)
        self.task_record = TaskRecord()
        self.extra_info = ExtraInfo(self.pa)


class Task:
    def __init__(self, res_vec, task_len, task_id, enter_time):
        self.id = task_id
        self.res_vec = res_vec
        self.len = task_len
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1


class TaskSlot:
    def __init__(self, pa):
        self.slot = [None] * pa.num_nw


class TaskBacklog:
    def __init__(self, pa):
        self.backlog = [None] * pa.backlog_size
        self.curr_size = 0


class TaskRecord:
    def __init__(self):
        self.record = {}


class MachineCluster:
    def __init__(self, pa):
        self.num_res = pa.num_res
        self.time_horizon = pa.time_horizon
        machine1, machine2, machine3 = Machine(0, 0), Machine(0, 0), Machine(1, 0)
        self.machine_cluster = [machine1, machine2, machine3]
        self.res_slot = 0
        for machine in self.machine_cluster:
            self.res_slot += machine.res_slot

        self.avbl_slot = np.ones((self.time_horizon, self.num_res)) * self.res_slot

        self.running_task = []

        # colormap for graphical representation
        self.colormap = np.arange(1 / float(pa.job_num_cap), 1, 1 / float(pa.job_num_cap))
        np.random.shuffle(self.colormap)

        # graphical representation
        self.canvas = np.zeros((self.num_res, self.time_horizon, self.res_slot))

    def allocate_resource(self, mtype, ope, time):
        new_res_slot = self.res_slot
        if ope < 0:
            for i in range(len(self.machine_cluster)):
                if ope >=0:
                    break
                if self.machine_cluster[i].type == mtype:
                    new_avbl_slot = self.avbl_slot[:, :] - self.machine_cluster[i].res_slot
                    if np.all(new_avbl_slot[:] >= 0):
                        self.avbl_slot[:, :] = new_avbl_slot
                        new_res_slot -= self.machine_cluster[i].res_slot
                        self.machine_cluster[i] = None
                        ope += 1
            new_machine_cluster = []
            for m in self.machine_cluster:
                if m is not None:
                    new_machine_cluster.append(m)
            self.machine_cluster = new_machine_cluster

        else:
            for i in range(ope):
                m = Machine(mtype, time)
                self.machine_cluster.append(m)
                self.avbl_slot[:, :] += m.res_slot
                new_res_slot += m.res_slot

        new_canvas = np.zeros((self.num_res, self.time_horizon, new_res_slot))
        if new_res_slot <= self.res_slot:
            new_canvas = self.canvas[:, :, : new_res_slot]
        else:
            new_canvas[:, :, : self.res_slot] = self.canvas
        self.canvas = new_canvas
        self.res_slot = new_res_slot

    def allocate_task(self, task, curr_time):

        allocated = False

        for t in range(0, self.time_horizon - task.len):

            new_avbl_res = self.avbl_slot[t: t + task.len, :] - task.res_vec

            if np.all(new_avbl_res[:] >= 0):

                allocated = True

                self.avbl_slot[t: t + task.len, :] = new_avbl_res
                task.start_time = curr_time + t
                task.finish_time = task.start_time + task.len

                self.running_task.append(task)

                # update graphical representation
                used_color = np.unique(self.canvas[:])
                # WARNING: there should be enough colors in the color map
                for color in self.colormap:
                    if color not in used_color:
                        new_color = color
                        break

                assert task.start_time != -1
                assert task.finish_time != -1
                assert task.finish_time > task.start_time
                canvas_start_time = task.start_time - curr_time
                canvas_end_time = task.finish_time - curr_time

                for res in range(self.num_res):
                    for i in range(canvas_start_time, canvas_end_time):
                        avbl_slot = np.where(self.canvas[res, i, :] == 0)[0]
                        self.canvas[res, i, avbl_slot[: task.res_vec[res]]] = new_color

                break

        return allocated

    def time_proceed(self, curr_time):

        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
        self.avbl_slot[-1, :] = self.res_slot

        for task in self.running_task:

            if task.finish_time <= curr_time:
                self.running_task.remove(task)

        # update graphical representation
        self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
        self.canvas[:, -1, :] = 0

class Machine:
    def __init__(self, mtype=0, time=0):
        self.type = mtype
        self.price_per_time = [0.2, 0.5][mtype]
        self.res_slot = [2, 5][mtype]
        self.last_account_time = time


class ExtraInfo:
    def __init__(self, pa):
        self.time_since_last_new_task = 0
        self.max_tracking_time_since_last_task = pa.max_track_since_new

    def new_task_comes(self):
        self.time_since_last_new_task = 0

    def time_proceed(self):
        if self.time_since_last_new_task < self.max_tracking_time_since_last_task:
            self.time_since_last_new_task += 1


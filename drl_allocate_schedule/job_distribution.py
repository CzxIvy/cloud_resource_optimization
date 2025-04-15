import numpy as np
import parameters

class Dist:

    def __init__(self, num_res, max_nw_size, task_len):
        self.num_res = num_res
        self.max_nw_size = max_nw_size
        self.task_len = task_len

        self.task_small_chance = 0.7

        self.task_len_big_lower = task_len * 2 / 3
        self.task_len_big_upper = task_len

        self.task_len_small_lower = 1
        self.task_len_small_upper = task_len / 5

        self.dominant_res_lower = max_nw_size / 2
        self.dominant_res_upper = max_nw_size

        self.other_res_lower = 1
        self.other_res_upper = max_nw_size / 5

    def normal_dist(self):

        # new work duration
        nw_len = np.random.randint(1, self.task_len + 1)  # same length in every dimension

        nw_size = np.zeros(self.num_res)

        for i in range(self.num_res):
            nw_size[i] = np.random.randint(1, self.max_nw_size + 1)

        return nw_len, nw_size

    def bi_model_dist(self):

        # -- job length --
        if np.random.rand() < self.task_small_chance:  # small job
            nw_len = np.random.randint(self.task_len_small_lower,
                                       self.task_len_small_upper + 1)
        else:  # big job
            nw_len = np.random.randint(self.task_len_big_lower,
                                       self.task_len_big_upper + 1)

        nw_size = np.zeros(self.num_res)

        # -- job resource request --
        dominant_res = np.random.randint(0, self.num_res)
        for i in range(self.num_res):
            if i == dominant_res:
                nw_size[i] = np.random.randint(self.dominant_res_lower,
                                               self.dominant_res_upper + 1)
            else:
                nw_size[i] = np.random.randint(self.other_res_lower,
                                               self.other_res_upper + 1)

        return nw_len, nw_size


def generate_sequence_work(pa, seed=42):

    np.random.seed(seed)
    nw_dist = pa.dist.bi_model_dist

    job_seqs = [None] * pa.num_ex
    for ex in range(pa.num_ex):
        job_simu_len = int(np.random.rand() * (pa.job_max_simu_len - pa.job_min_simu_len)) + pa.job_min_simu_len
        job_seq_per_ex = [None] * job_simu_len
        for i in range(job_simu_len):
            if np.random.rand() > pa.new_job_rate:
                continue
            task_simu_len = int(np.random.rand() * (pa.task_max_simu_len - pa.task_min_simu_len)) + pa.task_min_simu_len
            task_len_seq, task_size_seq = [], []
            for j in range(task_simu_len):
                task_len, task_size = nw_dist()
                task_len_seq.append(task_len)
                task_size_seq.append(task_size)
            task_len_seq = np.array(task_len_seq)
            task_size_seq = np.array(task_size_seq, dtype=int)
            job_seq_per_ex[i] = [task_len_seq, task_size_seq]
        job_seqs[ex] = job_seq_per_ex

    return job_seqs


if __name__ == '__main__':
    pa = parameters.Parameters()
    job_seqs = generate_sequence_work(pa)
    print(job_seqs)
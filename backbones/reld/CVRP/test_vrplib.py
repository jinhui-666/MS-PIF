import vrplib
import numpy as np
import torch
import yaml
import json
import time
import os
from pathlib import Path
from datetime import datetime
from CVRPModel import CVRPModel
from CVRPEnv import CVRPEnv
from utils import rollout, check_feasible, CVRPLib_XL_BKS
import random


BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR.parents[2] / "dataset"


def rollout(model, env, eval_type='greedy'):
    env.reset()
    actions = []
    probs = []
    reward = None
    state, reward, done = env.pre_step()

    while not done:
        cur_dist = env.get_cur_feature()
        selected, one_step_prob = model.one_step_rollout(state, cur_dist, eval_type=eval_type)
        # selected, one_step_prob = model(state)
        state, reward, done = env.step(selected)
        actions.append(selected)
        probs.append(one_step_prob)

    actions = torch.stack(actions, 1)
    if eval_type == 'greedy':
        probs = None
    else:
        probs = torch.stack(probs, 1)

    return torch.transpose(actions, 1, 2), probs, reward


class VRPLib_Tester:

    def __init__(self, config):
        self.config = config
        model_params = config['model_params']
        load_checkpoint = BASE_DIR / config['load_checkpoint']
        self.multiple_width = config['test_params']['pomo_size']
        self.cvrplib_xl_bks = CVRPLib_XL_BKS

        # cuda
        USE_CUDA = config['use_cuda']
        if USE_CUDA:
            cuda_device_num = config['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        
        # load trained model
        self.model = CVRPModel(**model_params)
        checkpoint = torch.load(load_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if config['vrplib_set'] == 'X':
            self.vrplib_path = DATASET_ROOT / 'CVRPLib-Set-X'
        elif config['vrplib_set'] == 'XL':
            self.vrplib_path = DATASET_ROOT / 'Vrp-Set-XL'
        elif config['vrplib_set'] == 'X_XXL_Li':
            self.vrplib_path = DATASET_ROOT / 'Vrp-Set-X_XXL_Li_ge1000'
        else:
            raise ValueError(f"Unknown vrplib_set: {config['vrplib_set']}")
        # self.vrplib_path = 'data/VRPLib/Vrp-Set-X' if config['vrplib_set'] == 'X' else "data/VRPLib/Vrp-Set-XXL"
        self.repeat_times = 1
        self.aug_factor = config['test_params']['aug_factor']
        print("AUG_FACTOR: ", self.aug_factor)
        self.vrplib_results = None
        
    def test_on_vrplib(self):

        output_dir = BASE_DIR / "result/test"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"test_vrplib_{timestamp}.txt"
        file_path = output_dir / file_name

        files = sorted(os.listdir(self.vrplib_path))
        vrplib_results = [] # [problem_size, elapsed_time, cost, gap]
        total_time = 0.

        for name in files:
            elapsed_time_all, best_cost_all, gap_all = [], [], []
            if '.sol' in name:
                continue
            name = name[:-4]
            instance_file = self.vrplib_path / f"{name}.vrp"
            solution_file = self.vrplib_path / f"{name}.sol"
            if solution_file.exists():
                solution = vrplib.read_solution(str(solution_file))
                optimal = solution['cost']
            else:
                optimal = self.cvrplib_xl_bks[name]

            for t in range(self.repeat_times):
                problem_size, elapsed_time, best_cost, gap = self.test_on_one_ins(name=name, instance=instance_file, optimal=optimal)
                elapsed_time_all.append(elapsed_time)
                best_cost_all.append(best_cost)
                gap_all.append(gap)

            elapsed_time_avg = np.mean(elapsed_time_all)
            cost_avg = np.mean(best_cost_all)
            gap_avg = np.mean(gap_all)

            output_str = f"Instance {name}: Time {elapsed_time_avg:.4f}s, Cost {cost_avg}, Gap {gap_avg:.5f}"
            print(output_str)
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(output_str + '\n')

            total_time += elapsed_time_avg
            vrplib_results.append([problem_size, elapsed_time_avg, cost_avg, gap_avg])


        if 'X_XXL_Li' in str(self.vrplib_path):

            gaps = [values[3] for values in vrplib_results]
            elapsed_time = [values[1] for values in vrplib_results]
            average_gap = np.mean(gaps) * 100
            average_time = np.mean(elapsed_time)

            print(f"X_XXL_Li Gap mean: {average_gap:.3f}%")
            print(f"X_XXL_Li Elapsed_time mean: {average_time:.4f}")

        elif 'XL' in str(self.vrplib_path):
            g1, g2, g3 = [], [], []
            total = []

            for size, _, _, gap in vrplib_results:
                if size <= 2000:
                    g1.append(gap)
                elif size <= 5000:
                    g2.append(gap)
                elif size <= 10000:
                    g3.append(gap)

                total.append(gap)

            print("Average gap [1000, 2000]: {:.3f}%".format(100 * (np.array(g1).mean())))
            print("Average gap (2000, 5000]: {:.3f}%".format(100 * (np.array(g2).mean())))
            print("Average gap (5000, 10000]: {:.3f}%".format(100 * (np.array(g3).mean())))
            print("Average gap total: {:.3f}%".format(100 * (np.array(total).mean())))
            print("Average time: {:.4f}s".format(total_time / len(vrplib_results)))

        else:
            g1, g2, g3 = [], [], []
            total = []

            for size, _, _, gap in vrplib_results:
                if size <= 200:
                    g1.append(gap)
                elif size <= 500:
                    g2.append(gap)
                elif size <= 1000:
                    g3.append(gap)

                total.append(gap)

            print("Average gap [1, 200]: {:.3f}%".format(100 * (np.array(g1).mean())))
            print("Average gap (200, 500]: {:.3f}%".format(100 * (np.array(g2).mean())))
            print("Average gap (500, 1000]: {:.3f}%".format(100 * (np.array(g3).mean())))
            print("Average gap total: {:.3f}%".format(100 * (np.array(total).mean())))
            print("Average time: {:.4f}s".format(total_time / len(vrplib_results)))



    def test_on_one_ins(self, name, instance, optimal):

        start_time = time.time()
        instance = vrplib.read_instance(str(instance))
        problem_size = instance['node_coord'].shape[0] - 1
        multiple_width = min(problem_size, self.multiple_width)

        # Initialize CVRP state
        env = CVRPEnv(multiple_width, self.device)

        aug_reward = None
        sep_augmentation = False
        if sep_augmentation:
            # compute only one augmented version each time to save gpu memory, repeat 8 times for each instance
            for idx in range(8):
                env.load_vrplib_problem(instance, aug_factor=self.aug_factor, aug_idx=idx)

                reset_state, reward, done = env.reset()
                self.model.eval()
                self.model.requires_grad_(False)
                self.model.pre_forward(reset_state)

                with torch.no_grad():
                    policy_solutions, policy_prob, rewards = rollout(self.model, env, 'greedy')
                # Return
                
                if aug_reward is not None:
                    aug_reward = rewards.reshape(self.aug_factor, 1, env.multi_width)
                    # shape: (augmentation, batch, multi)
                else:
                    aug_reward = rewards.reshape(1, 1, env.multi_width)

        else:
            env.load_vrplib_problem(instance, aug_factor=self.aug_factor, aug_idx=-1)

            reset_state, reward, done = env.reset()
            self.model.eval()
            self.model.requires_grad_(False)
            self.model.pre_forward(reset_state)

            with torch.no_grad():
                policy_solutions, policy_prob, rewards = rollout(self.model, env, 'greedy')

            aug_reward = rewards.reshape(self.aug_factor, 1, env.multi_width)
            # shape: (augmentation, batch, multi)    
        end_time = time.time()

        # shape: (augmentation, batch, multi)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_cost = -max_aug_pomo_reward.float()  # negative sign to make positive value
        best_cost = aug_cost.cpu().numpy().tolist()[0]

        elapsed_time = end_time - start_time

        gap = (best_cost - optimal) / optimal

        return problem_size, elapsed_time, best_cost, gap




if __name__ == "__main__":
    with open(BASE_DIR / 'config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    tester = VRPLib_Tester(config=config)
    tester.test_on_vrplib()

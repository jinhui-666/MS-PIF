import torch

import random

import time
import sys
import os
from datetime import datetime
import logging
import logging.config
import pytz
import numpy as np
import matplotlib.pyplot as plt
import json
import shutil

def rollout(model, env, eval_type='greedy'):
    env.reset()
    actions = []
    probs = []
    reward = None
    state, reward, done = env.pre_step()

    t = 0
    while not done:
        cur_dist = env.get_cur_feature()
        selected, one_step_prob = model.one_step_rollout(state, cur_dist, eval_type=eval_type)
        state, reward, done = env.step(selected)

        actions.append(selected)
        probs.append(one_step_prob)
        t += 1

    actions = torch.stack(actions, 1)
    if eval_type == 'greedy':
        probs = None
    else:
        probs = torch.stack(probs, 1)

    return torch.transpose(actions, 1, 2), probs, reward

def batched_two_opt_torch(cuda_points, cuda_tour, max_iterations=1000, device="cpu"):
  cuda_tour = torch.cat((cuda_tour, cuda_tour[:, 0:1]), dim=-1)
  iterator = 0
  problem_size = cuda_points.shape[0]
  with torch.inference_mode():
    batch_size = cuda_tour.shape[0]
    min_change = -1.0
    while min_change < 0.0:
      points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
      points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))

      A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
      A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
      A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
      A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

      change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
      valid_change = torch.triu(change, diagonal=2)

      min_change = torch.min(valid_change)
      flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
      min_i = torch.div(flatten_argmin_index, problem_size, rounding_mode='floor')
      min_j = torch.remainder(flatten_argmin_index, problem_size)

      if min_change < -1e-6:
        for i in range(batch_size):
          cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
        iterator += 1
      else:
        break

      if iterator >= max_iterations:
        break

  return cuda_tour[:, :-1]

def augment_xy_data_by_8_fold(problems, aug_idx=-1):
    # problems.shape: (batch, problem, 2)
    if aug_idx==-1:
        x = problems[:, :, [0]]
        y = problems[:, :, [1]]
        # x,y shape: (batch, problem, 1)

        dat1 = torch.cat((x, y), dim=2)
        dat2 = torch.cat((1 - x, y), dim=2)
        dat3 = torch.cat((x, 1 - y), dim=2)
        dat4 = torch.cat((1 - x, 1 - y), dim=2)
        dat5 = torch.cat((y, x), dim=2)
        dat6 = torch.cat((1 - y, x), dim=2)
        dat7 = torch.cat((y, 1 - x), dim=2)
        dat8 = torch.cat((1 - y, 1 - x), dim=2)

        aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
        # shape: (8*batch, problem, 2)
    else:
        x = problems[:, :, [0]]
        y = problems[:, :, [1]]
        # x,y shape: (batch, problem, 1)
        
        aug_x = x
        aug_y = y

        if aug_idx >= 4:
            aug_x = y
            aug_y = x

        if aug_idx % 2 == 1:
            aug_x = 1 - aug_x

        if aug_idx % 4 >= 2:
            aug_y = 1 - aug_y
      
        aug_problems = torch.cat((aug_x, aug_y), dim=-1)
        # shape: (batch, problem, 2)

    return aug_problems

def check_feasible(pi, demand):
  # input shape: (1, multi, problem) 
  pi = pi.squeeze(0)
  multi = pi.shape[0]
  problem_size = demand.shape[1]
  demand = demand.expand(multi, problem_size)
  sorted_pi = pi.data.sort(1)[0]

  # Sorting it should give all zeros at front and then 1...n
  assert (
      torch.arange(1, problem_size + 1, out=pi.data.new()).view(1, -1).expand(multi, problem_size) ==
      sorted_pi[:, -problem_size:]
  ).all() and (sorted_pi[:, :-problem_size] == 0).all(), "Invalid tour"

  # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
  demand_with_depot = torch.cat(
      (
          torch.full_like(demand[:, :1], -1),
          demand
      ),
      1
  )
  d = demand_with_depot.gather(1, pi)

  used_cap = torch.zeros_like(demand[:, 0])
  for i in range(pi.size(1)):
      used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
      # Cannot use less than 0
      used_cap[used_cap < 0] = 0
      assert (used_cap <= 1 + 1e-4).all(), "Used more than capacity"

class Logger(object):
  def __init__(self, filename, config):
    '''
    filename: a json file
    '''
    self.filename = filename
    self.logger = config
    self.logger['result'] = {}
    self.logger['result']['val_100'] = []
    self.logger['result']['val_200'] = []
    self.logger['result']['val_500'] = []

  def log(self, info):
    '''
    Log validation cost on 3 datasets every log step
    '''
    self.logger['result']['val_100'].append(info[0])
    self.logger['result']['val_200'].append(info[1])
    self.logger['result']['val_500'].append(info[2])

    with open(self.filename, 'w') as f:
      json.dump(self.logger, f)

process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))
result_folder = './result/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'

def get_result_folder():
    return result_folder

def set_result_folder(folder):
    global result_folder
    result_folder = folder

def create_logger(log_file=None):
    if 'filepath' not in log_file:
        log_file['filepath'] = get_result_folder()

    if 'desc' in log_file:
        log_file['filepath'] = log_file['filepath'].format(desc='_' + log_file['desc'])
    else:
        log_file['filepath'] = log_file['filepath'].format(desc='')

    set_result_folder(log_file['filepath'])

    if 'filename' in log_file:
        filename = log_file['filepath'] + '/' + log_file['filename']
    else:
        filename = log_file['filepath'] + '/' + 'log.txt'

    if not os.path.exists(log_file['filepath']):
        os.makedirs(log_file['filepath'])

    file_mode = 'a' if os.path.isfile(filename)  else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


class LogData:
    def __init__(self):
        self.keys = set()
        self.data = {}

    def get_raw_data(self):
        return self.keys, self.data

    def set_raw_data(self, r_data):
        self.keys, self.data = r_data

    def append_all(self, key, *args):
        if len(args) == 1:
            value = [list(range(len(args[0]))), args[0]]
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].extend(value)
        else:
            self.data[key] = np.stack(value, axis=1).tolist()
            self.keys.add(key)

    def append(self, key, *args):
        if len(args) == 1:
            args = args[0]

            if isinstance(args, int) or isinstance(args, float):
                if self.has_key(key):
                    value = [len(self.data[key]), args]
                else:
                    value = [0, args]
            elif type(args) == tuple:
                value = list(args)
            elif type(args) == list:
                value = args
            else:
                raise ValueError('Unsupported value type')
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].append(value)
        else:
            self.data[key] = [value]
            self.keys.add(key)

    def get_last(self, key):
        if not self.has_key(key):
            return None
        return self.data[key][-1]

    def has_key(self, key):
        return key in self.keys

    def get(self, key):
        split = np.hsplit(np.array(self.data[key]), 2)

        return split[1].squeeze().tolist()

    def getXY(self, key, start_idx=0):
        split = np.hsplit(np.array(self.data[key]), 2)

        xs = split[0].squeeze().tolist()
        ys = split[1].squeeze().tolist()

        if type(xs) is not list:
            return xs, ys

        if start_idx == 0:
            return xs, ys
        elif start_idx in xs:
            idx = xs.index(start_idx)
            return xs[idx:], ys[idx:]
        else:
            raise KeyError('no start_idx value in X axis data.')

    def get_keys(self):
        return self.keys


class TimeEstimator:
    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count-1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time*60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time*60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            count, total, elapsed_time_str, remain_time_str))


def util_print_log_array(logger, result_log: LogData):
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    for key in result_log.get_keys():
        logger.info('{} = {}'.format(key+'_list', result_log.get(key)))


def util_save_log_image_with_label(result_file_prefix,
                                   img_params,
                                   result_log: LogData,
                                   labels=None):
    dirname = os.path.dirname(result_file_prefix)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    _build_log_image_plt(img_params, result_log, labels)

    if labels is None:
        labels = result_log.get_keys()
    file_name = '_'.join(labels)
    fig = plt.gcf()
    fig.savefig('{}-{}.jpg'.format(result_file_prefix, file_name))
    plt.close(fig)


def _build_log_image_plt(img_params,
                         result_log: LogData,
                         labels=None):
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    # Read json
    folder_name = img_params['json_foldername']
    file_name = img_params['filename']
    log_image_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name, file_name)

    with open(log_image_config_file, 'r') as f:
        config = json.load(f)

    figsize = (config['figsize']['x'], config['figsize']['y'])
    plt.figure(figsize=figsize)

    if labels is None:
        labels = result_log.get_keys()
    for label in labels:
        plt.plot(*result_log.getXY(label), label=label)

    ylim_min = config['ylim']['min']
    ylim_max = config['ylim']['max']
    if ylim_min is None:
        ylim_min = plt.gca().dataLim.ymin
    if ylim_max is None:
        ylim_max = plt.gca().dataLim.ymax
    plt.ylim(ylim_min, ylim_max)

    xlim_min = config['xlim']['min']
    xlim_max = config['xlim']['max']
    if xlim_min is None:
        xlim_min = plt.gca().dataLim.xmin
    if xlim_max is None:
        xlim_max = plt.gca().dataLim.xmax
    plt.xlim(xlim_min, xlim_max)

    plt.rc('legend', **{'fontsize': 18})
    plt.legend()
    plt.grid(config["grid"])


def copy_all_src(dst_root):
    # execution dir
    if os.path.basename(sys.argv[0]).startswith('ipykernel_launcher'):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(sys.argv[0])

    # home dir setting
    tmp_dir1 = os.path.abspath(os.path.join(execution_path, sys.path[0]))
    tmp_dir2 = os.path.abspath(os.path.join(execution_path, sys.path[1]))

    if len(tmp_dir1) > len(tmp_dir2) and os.path.exists(tmp_dir2):
        home_dir = tmp_dir2
    else:
        home_dir = tmp_dir1

    # make target directory
    dst_path = os.path.join(dst_root, 'src')

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for item in sys.modules.items():
        key, value = item

        if hasattr(value, '__file__') and value.__file__:
            src_abspath = os.path.abspath(value.__file__)

            if os.path.commonprefix([home_dir, src_abspath]) == home_dir:
                dst_filepath = os.path.join(dst_path, os.path.basename(src_abspath))

                if os.path.exists(dst_filepath):
                    split = list(os.path.splitext(dst_filepath))
                    split.insert(1, '({})')
                    filepath = ''.join(split)
                    post_index = 0

                    while os.path.exists(filepath.format(post_index)):
                        post_index += 1

                    dst_filepath = filepath.format(post_index)

                shutil.copy(src_abspath, dst_filepath)



def plot_att_score(path, step, cur_xy, selected_xy, coord_xy, score):
    sz = len(coord_xy)
    idx = [i for i in range(sz)]

    def l2(coord1, coord2):
        return (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2

    def cmp(x):
        return l2(cur_xy, coord_xy[x])
        return (cur_xy[0] - coord_xy[x][0]) ** 2 + (cur_xy[1] - coord_xy[x][1]) ** 2

    idx.sort(key=cmp)

    score = score[idx]
    # print(len(score.tolist()))
    x_values = [x for x in range(len(score))]
    # print("Num Remaining: ", score.shape)

    plt.clf()
    plt.scatter(x_values, score.tolist(), s=15, marker='x')

    plt.xlabel('Index (ascending distance)')
    plt.ylabel('Att Score')
    plt.title('Remaining Nodes: {}'.format(len(score)))

    # Saving the plot
    plt.savefig(path + '{}.png'.format(step))

    # plt.show()

def plot_scatter_score(path, step, cur_xy, selected_xy, coord_xy, score, depot_xy=None):
    if score == None or len(score) == 0:
        return
    plt.clf()

    # Separate the coordinates and scores into lists
    x_values = [point[0] for point in coord_xy]
    y_values = [point[1] for point in coord_xy]

    score = score.detach().cpu()
    score_min = min(score)
    score_max = max(score)
    # print(score_max, score_min)
    normalized_scores = [(score - score_min) / (score_max - score_min + 0.00001) * 0.7 + 0.25 for score in score]

    import matplotlib.colors as mcolors

    # plot the current node
    cur = plt.scatter(cur_xy[0], cur_xy[1], color='red', marker='*', s=50, label='Center')
    nxt = plt.scatter(selected_xy[0], selected_xy[1], color='None', marker='^', s=50, label='Center', edgecolors='r')
    if depot_xy != None:
        dpt = plt.scatter(depot_xy[0], depot_xy[1], color='orange', marker='o', s=50, label='Center')
        if cur_xy[0] == depot_xy[0] and cur_xy[1] == depot_xy[1]:
            plt.legend((dpt, cur, nxt), ('Depot', 'Current at depot', 'Next'), loc='best')
        else:
            plt.legend((dpt, cur, nxt), ('Depot', 'Current', 'Next'), loc='best')
    else:
        plt.legend((cur, nxt), ('Current', 'Next'), loc='best')

    colorbar = plt.colorbar(label='Score')  # Add colorbar to show the mapping between colors and scores
    # colorbar.ax.invert_yaxis()

    # Create a custom colormap with colors that accentuate larger scores
    colors = plt.cm.viridis(np.linspace(1, 0, 256))  # Get colors from viridis colormap
    # colors[:, :3] *= 1.5  # Increase intensity of colors to emphasize larger scores
    custom_cmap = mcolors.ListedColormap(colors)
    # Plot the points with associated scores
    plt.scatter(x_values, y_values, c=score.cpu().tolist(), cmap=custom_cmap, s=10, alpha=normalized_scores)



    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('{} Nodes with Associated Scores'.format(len(score)))
    # plt.title('Remaining Nodes: {}'.format(len(score)))

    # Saving the plot
    plt.savefig(path + 'scatter_{}.png'.format(step))
    plt.close("all")

    # plt.show()


def plot_gap(path, step_list, gap):
    plt.clf()
    # plt.style.use('ggplot')
    plt.figure(figsize=(10, 5))
    plt.title("Gap of greedy w.r.t. model output")
    plt.xlabel("Number of selected nodes")
    # plt.xticks(rotation=45)
    plt.ylabel("Gap (%)")
    plt.plot(step_list, gap, c='red')
    # plt.plot(step_list, gap, 'ro-')
    plt.scatter(step_list, gap, marker='o', c='red', s=10)
    plt.legend()

    plt.savefig(path + 'Gap_Plot.png')
    plt.close("all")


def plot_line_chart(path, data, title, ylabel, xdata=None):
    plt.clf()
    # plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Epoch")
    # plt.xticks(rotation=45)
    plt.ylabel(ylabel)
    n = len(data)
    if xdata == None:
        xdata = range(n)
    plt.plot(xdata, data, 'bo-')
    # plt.plot(step_list, gap, 'ro-')
    # plt.scatter(step_list, gap, marker='o', c='red', s=10)
    # plt.legend()

    plt.savefig(path + '-' + title + str(n) + '.png')
    plt.close("all")

def plot_bar_chart(path, data, title, ylabel, xdata=None):
    plt.clf()
    # plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylim(0.0, 1.0)
    # plt.xticks(rotation=45)
    plt.ylabel(ylabel)
    n = len(data)
    if xdata == None:
        xdata = range(n)
    plt.bar(xdata, data)
    # plt.plot(step_list, gap, 'ro-')
    # plt.scatter(step_list, gap, marker='o', c='red', s=10)
    # plt.legend()

    plt.savefig(path + '-' + title + str(n) + '.png')
    plt.close("all")

CVRPLib_XL_BKS = {
    "XL-n1048-k237": 380211,
    "XL-n1094-k157": 112431,
    "XL-n1141-k112": 95727,
    "XL-n1188-k96": 104415,
    "XL-n1234-k55": 96647,
    "XL-n1281-k29": 31101,
    "XL-n1328-k19": 38247,
    "XL-n1374-k278": 233049,
    "XL-n1421-k232": 384826,
    "XL-n1468-k151": 250166,
    "XL-n1514-k106": 92425,
    "XL-n1561-k75": 101549,
    "XL-n1608-k39": 48021,
    "XL-n1654-k11": 36385,
    "XL-n1701-k562": 521136,
    "XL-n1748-k271": 173896,
    "XL-n1794-k163": 141729,
    "XL-n1841-k126": 214038,
    "XL-n1888-k82": 143623,
    "XL-n1934-k46": 53013,
    "XL-n1981-k13": 32580,
    "XL-n2028-k617": 544403,
    "XL-n2074-k264": 421627,
    "XL-n2121-k186": 283211,
    "XL-n2168-k138": 127298,
    "XL-n2214-k131": 154676,
    "XL-n2261-k54": 98907,
    "XL-n2307-k34": 47958,
    "XL-n2354-k631": 940825,
    "XL-n2401-k408": 463473,
    "XL-n2447-k290": 218706,
    "XL-n2494-k194": 361205,
    "XL-n2541-k121": 146390,
    "XL-n2587-k66": 73394,
    "XL-n2634-k17": 31641,
    "XL-n2681-k540": 798603,
    "XL-n2727-k546": 431134,
    "XL-n2774-k286": 407847,
    "XL-n2821-k208": 216763,
    "XL-n2867-k120": 165990,
    "XL-n2914-k95": 88990,
    "XL-n2961-k55": 108084,
    "XL-n3007-k658": 522319,
    "XL-n3054-k461": 782739,
    "XL-n3101-k311": 245937,
    "XL-n3147-k232": 256626,
    "XL-n3194-k161": 148728,
    "XL-n3241-k115": 221370,
    "XL-n3287-k30": 40229,
    "XL-n3334-k934": 1452698,
    "XL-n3408-k524": 678643,
    "XL-n3484-k436": 703355,
    "XL-n3561-k229": 209386,
    "XL-n3640-k211": 189724,
    "XL-n3721-k77": 162862,
    "XL-n3804-k29": 52885,
    "XL-n3888-k1010": 1880368,
    "XL-n3975-k687": 525901,
    "XL-n4063-k347": 548931,
    "XL-n4153-k291": 356034,
    "XL-n4245-k203": 229659,
    "XL-n4340-k148": 244226,
    "XL-n4436-k48": 61477,
    "XL-n4535-k1134": 1203566,
    "XL-n4635-k790": 610650,
    "XL-n4738-k487": 760501,
    "XL-n4844-k321": 404652,
    "XL-n4951-k203": 285269,
    "XL-n5061-k184": 161629,
    "XL-n5174-k55": 61382,
    "XL-n5288-k1246": 1960101,
    "XL-n5406-k783": 1040536,
    "XL-n5526-k553": 336898,
    "XL-n5649-k401": 644866,
    "XL-n5774-k290": 250207,
    "XL-n5902-k122": 217447,
    "XL-n6034-k61": 64448,
    "XL-n6168-k1922": 1530010,
    "XL-n6305-k1042": 1177528,
    "XL-n6445-k628": 996623,
    "XL-n6588-k473": 334068,
    "XL-n6734-k330": 448031,
    "XL-n6884-k148": 181809,
    "XL-n7037-k38": 70845,
    "XL-n7193-k1683": 2958979,
    "XL-n7353-k1471": 1537811,
    "XL-n7516-k859": 573902,
    "XL-n7683-k602": 702098,
    "XL-n7854-k365": 659221,
    "XL-n8028-k294": 266900,
    "XL-n8207-k108": 118274,
    "XL-n8389-k2028": 3358731,
    "XL-n8575-k1297": 1089137,
    "XL-n8766-k1032": 906406,
    "XL-n8960-k634": 773383,
    "XL-n9160-k379": 324092,
    "XL-n9363-k209": 205575,
    "XL-n9571-k55": 106791,
    "XL-n9784-k2774": 4078217,
    "XL-n10001-k1570": 2333757
}

import argparse
import csv
import tensorflow as tf
from keras import backend as K
import random

from config import get_defaults_cfg, setup_cfg

from controller import Controller, StateSpace
from manager import RewardManager
from matcher import build_matcher
from scenario_pool import build_pool

import shutil
import os


def remove_history_files(cfg):
    files = [cfg.TRAIN.HISTORY_FILE, cfg.TRAIN.BUFFER_FILE]

    if os.path.exists(cfg.CONTROLLER.WEIFHTS_DIR_PATH):
        shutil.rmtree(cfg.CONTROLLER.WEIFHTS_DIR_PATH)
    for file in files:
        if os.path.exists(file):
            os.remove(file)


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        # default="./config/SEMI_APOLLOSCAPE_VGG_nohaze_YOLO_PEOXY_RERANK.yaml",
        # default="./config/SEMI_APOLLOSCAPE_CODE_nohaze_YOLO_PEOXY_RERANK.yaml",
        # default="./config/SEMI_REPEAT_APOLLOSCAPE_VGG_nohaze_YOLO_PEOXY_RERANK.yaml",
        # default="./config/SEMI_REPEAT_APOLLOSCAPE_CODE_nohaze_YOLO_PEOXY_RERANK.yaml",

        # default="./config/SEMI_MANUAL_APOLLOSCAPE_VGG_nohaze_YOLO_PEOXY_RERANK.yaml",
        # default="./config/SEMI_MANUAL_APOLLOSCAPE_CODE_nohaze_YOLO_PEOXY_RERANK.yaml",
        default="./config/SEMI_MANUAL_REPEAT_APOLLOSCAPE_VGG_nohaze_YOLO_PEOXY_RERANK.yaml",
        # default="./config/SEMI_MANUAL_REPEAT_APOLLOSCAPE_CODE_nohaze_YOLO_PEOXY_RERANK.yaml",

        # default="./config/APOLLOSCAPE_VGG_nohaze_YOLO_KNN.yaml",
        # default="./config/APOLLOSCAPE_CODE_nohaze_YOLO_KNN.yaml",

        # default="./config/APOLLOSCAPE_VGG_haze_YOLO_KNN.yaml",
        # default="./config/APOLLOSCAPE_CODE_haze_YOLO_KNN.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # init args
    cfg = get_defaults_cfg()
    cfg = setup_cfg(cfg, args.cfg, args.opts)

    # init some buffers
    ini_state = []
    final_state = []
    ini_action = []
    final_action = []

    # load the scenario_pool
    scenario_pool = build_pool(cfg)

    # create a shared session between Keras and Tensorflow
    policy_sess = tf.Session()
    K.set_session(policy_sess)

    # clear the previous files
    remove_history_files(cfg)

    # construct a state space
    state_space = StateSpace()
    for step in range(cfg.SEARCHSPACE.NUM_LAYERS):
        state_space.add_state(name=cfg.SEARCHSPACE.LAYER_NAMES[step], values=cfg.SEARCHSPACE.LAYER_VALUES[step])

    # print the state space being searched
    # state_space.print_state_space()
    previous_res_f1 = 0.0
    total_reward = 0.0

    with policy_sess.as_default():
        # create the Controller and build the internal policy network
        controller = Controller(policy_sess, cfg, state_space)

    # create the Manager
    manager = RewardManager(cfg=cfg)
    # create the Matcher
    matcher = build_matcher(cfg)

    # get an initial random state space if controller needs to predict an
    # action from the initial state
    state = state_space.get_random_state_space(cfg.CONTROLLER.NUM_LAYERS)
    print("Initial Random State : ", state_space.parse_state_space_list(state))
    print()

    # ##########################
    # Split the dataset
    # ##########################

    # get unlabeled data index
    datasetSize = cfg.MANAGER.DATASET.SIZE
    assert cfg.MANAGER.DATASET.MODE in ['RANDOM', 'MANUAL']
    if cfg.MANAGER.DATASET.MODE == 'RANDOM': 
        # random(随机划分)
        semi_list = random.sample(range(0, datasetSize), int(datasetSize / 2))
        # semi_list = [2 * i for i in range(int(datasetSize / 2))]
    else:
        # manual(人为划分)
        semi_list = [i for i in range(int(datasetSize / 2))]

    # ##########################
    # train for number of trails
    # ##########################
    for trial in range(cfg.CONTROLLER.MAX_TRIALS):

        # get an action that does not match the scenario pool
        while True:
            with policy_sess.as_default():
                K.set_session(policy_sess)
                actions = controller.get_action(state)  # get an action for the previous state

            # search data_index
            rank_index = matcher.get_index(state_space.parse_state_space_list(actions))

            # nearest neighbor fill
            # 1.Repeat Mode
            if cfg.MATCHER.SEMI_REPEAT:
                for i in range(len(rank_index)):
                    if rank_index[i] not in semi_list:
                        continue
                    nearest_index = matcher.get_nearest_repeat(semi_list, rank_index[i])
                    rank_index[i] = nearest_index
            # 2.Not Repeat Mode
            else:
                for i in range(len(rank_index)):
                    if (rank_index[i] not in semi_list) and (rank_index[i] not in rank_index[0:i]):
                        continue
                    nearest_index = matcher.get_nearest(semi_list, rank_index[0:i], rank_index[i])
                    rank_index[i] = nearest_index

            if scenario_pool.is_inpool(rank_index) or (not cfg.SNENARIOPOOL.FLAG):
                break

        # print the action probabilities
        # 可以注释掉提升速度
        # state_space.print_actions(actions)
        # print("Predicted actions : ", state_space.parse_state_space_list(actions))

        # get reward and res_map from the manager
        print("rank_index:", rank_index)
        reward, previous_res_f1, _, _, _ = manager.get_reward(rank_index)
        print("Rewards : ", reward, "Res_f1 : ", previous_res_f1)

        with policy_sess.as_default():
            K.set_session(policy_sess)

            total_reward += reward
            print("Total reward : ", total_reward)

            # actions and states are equivalent, save the state and reward
            state = actions
            controller.store_rollout(state, reward)

            # train the controller on the saved state and the discounted rewards
            loss = controller.train_step()
            print("Trial %d: Controller loss : %0.6f" % (trial + 1, loss))

            if cfg.TRAIN.HISTORY_STORE:
                with open(cfg.TRAIN.HISTORY_FILE, mode='a+') as f:
                    data = [total_reward, previous_res_f1, reward]
                    data.extend(state_space.parse_state_space_list(state))
                    writer = csv.writer(f)
                    writer.writerow(data)
        print()

    # get the final results and save it
    with policy_sess.as_default():
        K.set_session(policy_sess)
        actions = controller.get_action(state, explore_flag=False)
        rank_index = matcher.get_index(state_space.parse_state_space_list(actions))

        with open(cfg.CSV_SAVE_PATH, 'a', newline='') as f:
            data = []
            data.extend(state_space.parse_state_space_list(actions))
            writer = csv.writer(f)
            writer.writerow(data)

        # add the scenario to scenario_pool
        with open("./scenario_pool/scenario_pool.csv", 'a', newline='') as f:
            data = []
            data.extend(rank_index)
            writer = csv.writer(f)
            writer.writerow(data)

        print('final_state:', state_space.parse_state_space_list(actions))

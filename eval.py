import argparse
import csv

from config import get_defaults_cfg, setup_cfg

from manager import RewardManager
from matcher import build_matcher


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,

        # default="./config/EVB17_YOlO_KNN.yaml",
        # default="./config/EVB17_FRCNN_R50_C4_1x_KNN.yaml",
        # default="./config/EVB17_FRCNN_R50_C4_3x_KNN.yaml",
        # default="./config/EVB17_FRCNN_R50_DC5_3x_KNN.yaml",
        # default="./config/EVB17_FRCNN_R50_FPN_3x_KNN.yaml",
        # default="./config/EVB17_FRCNN_R101_C4_3x_KNN.yaml",
        # default="./config/EVB17_FRCNN_R101_DC5_3x_KNN.yaml",
        # default="./config/EVB17_FRCNN_R101_FPN_3x_KNN.yaml",
        # default="./config/EVB17_FRCNN_X101_32x8d_FPN_3x_KNN.yaml",
        # default="./config/EVB17_RetinaNet_R50_FPN_1x_KNN.yaml",
        # default="./config/EVB17_RetinaNet_R50_FPN_3x_KNN.yaml",
        # default="./config/EVB17_RetinaNet_R101_FPN_3x_KNN.yaml",

        # default="./config/BDD100K_TRAIN_YOLO_KNN.yaml",
        # default="./config/BDD100K_VAL_YOLO_KNN.yaml",

        # default="./config/KITTI_YOLO_KNN.yaml",
        # default="./config/KITTI_FRCNN_R50_C4_1x_KNN.yaml",
        # default="./config/KITTI_FRCNN_R50_FPN_3x_KNN.yaml",
        # default="./config/KITTI_FRCNN_R101_FPN_3x_KNN.yaml",
        # default="./config/KITTI_RetinaNet_R101_FPN_3x_KNN.yaml",

        # default="./config/APOLLOSCAPE_VGG_nohaze_YOLO_PEOXY_RERANK.yaml",
        # default="./config/APOLLOSCAPE_CODE_nohaze_YOLO_PEOXY_RERANK.yaml",

        # default="./config/APOLLOSCAPE_VGG_haze_YOLO_PEOXY_RERANK.yaml",
        # default="./config/APOLLOSCAPE_CODE_haze_YOLO_PEOXY_RERANK.yaml",

        # default="./config/APOLLOSCAPE_VGG_nohaze_YOLO_KNN.yaml",
        # default="./config/APOLLOSCAPE_CODE_nohaze_YOLO_KNN.yaml",

        # default="./config/APOLLOSCAPE_VGG_haze_YOLO_KNN.yaml",
        default="./config/APOLLOSCAPE_CODE_haze_YOLO_KNN.yaml",

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

    # create the Manager
    manager = RewardManager(cfg=cfg)

    # create the matcher and search data_index
    matcher = build_matcher(cfg)

    if cfg.EVAL.RS_FLAG:
        # all the scenario from results
        with open(cfg.EVAL.RS.RS_PATH, 'r') as f:
            reader = csv.reader(f)
            print(type(reader))
            for row in reader:
                rs_scene = list(map(float, row))
                rank_index = matcher.get_index(rs_scene)

                # get rec, prec, and f1 from the manager
                _, _, rec, prec, f1 = manager.get_reward(rank_index)
                print("Rec: ", rec, "Prec:", prec, "F1:", f1)

                with open(cfg.EVAL.RS.SAVE_PATH, mode='a+', newline='') as f:
                    data = rs_scene
                    data.extend([rec, prec, f1])
                    writer = csv.writer(f)
                    writer.writerow(data)

    if cfg.EVAL.GT_FLAG:
        # all the scenario from dataset
        for row in matcher.galScenario:
            rs_scene = list(map(float, row))
            rank_index = matcher.get_index(rs_scene)

            # get rec, prec, and f1 from the manager
            _, _, rec, prec, f1 = manager.get_reward(rank_index)
            print("Rec: ", rec, "Prec:", prec, "F1:", f1)
            print()

            with open(cfg.EVAL.GT.SAVE_PATH, mode='a+', newline='') as f:
                data = rs_scene
                data.extend([rec, prec, f1])
                writer = csv.writer(f)
                writer.writerow(data)

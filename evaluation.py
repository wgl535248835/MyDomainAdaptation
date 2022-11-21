import csv
import os
import time

import configargparse
import torch
from matplotlib import pyplot as plt
from numpy import mean
from sklearn.metrics import auc, roc_curve, average_precision_score, precision_recall_curve
from terminaltables import AsciiTable
from tqdm import tqdm

from main import get_model, load_data
from utils import str2bool, evaluate


# 进模型权重地址修改--resume_from,batch_size,transfer_loss
# python MyDA/evaluation.py --config MyDA/DAN/DAN.yaml --data_dir mydataset --src_domain state-farm --tgt_domain Real


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)

    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--tgt_domain', type=str, required=True)

    # training related
    parser.add_argument('--batch_size', type=int, default=32)  # 32
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False,
                        help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")
    parser.add_argument("--resume_from", type=str, default='tf-logs/Val_Epoch068-Acc55.960.pth',
                        help="the checkpoint file to resume from")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=10)
    parser.add_argument('--transfer_loss', type=str, default='mmd')
    return parser


def get_metrics_output(eval_results, metrics_output, classes_names, indexs, APs):
    f = open(metrics_output, 'a', newline='')
    writer = csv.writer(f)

    """
    输出并保存Accuracy、Precision、Recall、F1 Score、Confusion matrix结果
    """
    p_r_f1 = [['Classes', 'Precision', 'Recall', 'F1 Score', 'Average Precision']]
    for i in range(len(classes_names)):
        data = []
        data.append(classes_names[i])
        data.append('{:.2f}'.format(eval_results.get('precision')[indexs[i]]))
        data.append('{:.2f}'.format(eval_results.get('recall')[indexs[i]]))
        data.append('{:.2f}'.format(eval_results.get('f1_score')[indexs[i]]))
        data.append('{:.2f}'.format(APs[indexs[i]] * 100))
        p_r_f1.append(data)
    TITLE = 'Classes Results'
    TABLE_DATA_1 = tuple(p_r_f1)
    table_instance = AsciiTable(TABLE_DATA_1, TITLE)
    # table_instance.justify_columns[2] = 'right'
    print()
    print(table_instance.table)
    writer.writerows(TABLE_DATA_1)
    writer.writerow([])
    print()

    TITLE = 'Total Results'
    TABLE_DATA_2 = (
        ('Top-1 Acc', 'Top-5 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
        ('{:.2f}'.format(eval_results.get('accuracy_top-1', 0.0)),
         '{:.2f}'.format(eval_results.get('accuracy_top-5', 100.0)),
         '{:.2f}'.format(mean(eval_results.get('precision', 0.0))),
         '{:.2f}'.format(mean(eval_results.get('recall', 0.0))),
         '{:.2f}'.format(mean(eval_results.get('f1_score', 0.0)))),
    )
    table_instance = AsciiTable(TABLE_DATA_2, TITLE)
    # table_instance.justify_columns[2] = 'right'
    print(table_instance.table)
    writer.writerows(TABLE_DATA_2)
    writer.writerow([])
    print()

    writer_list = []
    writer_list.append([' '] + [str(c) for c in classes_names])
    for i in range(len(eval_results.get('confusion'))):
        writer_list.append([classes_names[i]] + [str(x) for x in eval_results.get('confusion')[i]])
    TITLE = 'Confusion Matrix'
    TABLE_DATA_3 = tuple(writer_list)
    table_instance = AsciiTable(TABLE_DATA_3, TITLE)
    print(table_instance.table)
    writer.writerows(TABLE_DATA_3)
    print()


# def get_prediction_output(preds, targets, image_paths, classes_names, indexs, prediction_output):
#     nums = len(preds)
#     f = open(prediction_output, 'a', newline='')
#     writer = csv.writer(f)
#
#     results = [['File', 'Pre_label', 'True_label', 'Success']]
#     results[0].extend(classes_names)
#
#     for i in range(nums):
#         temp = [image_paths[i]]
#         pred_label = classes_names[indexs[torch.argmax(preds[i]).item()]]
#         true_label = classes_names[indexs[targets[i].item()]]
#         success = True if pred_label == true_label else False
#         class_score = preds[i].tolist()
#         temp.extend([pred_label, true_label, success])
#         temp.extend(class_score)
#         results.append(temp)
#
#     writer.writerows(results)


def plot_ROC_curve(preds, targets, classes_names, savedir):
    rows = len(targets)
    cols = len(preds[0])
    ROC_output = os.path.join(savedir, 'ROC')
    PR_output = os.path.join(savedir, 'P-R')
    os.makedirs(ROC_output)
    os.makedirs(PR_output)
    APs = []
    for j in range(cols):
        gt, pre, pre_score = [], [], []
        for i in range(rows):
            if targets[i].item() == j:
                gt.append(1)
            else:
                gt.append(0)

            if torch.argmax(preds[i]).item() == j:
                pre.append(1)
            else:
                pre.append(0)

            pre_score.append(preds[i][j].item())

        # ROC
        ROC_csv_path = os.path.join(ROC_output, classes_names[j] + '.csv')
        ROC_img_path = os.path.join(ROC_output, classes_names[j] + '.png')
        ROC_f = open(ROC_csv_path, 'a', newline='')
        ROC_writer = csv.writer(ROC_f)
        ROC_results = []

        FPR, TPR, threshold = roc_curve(targets.tolist(), pre_score, pos_label=j)

        AUC = auc(FPR, TPR)

        ROC_results.append(['AUC', AUC])
        ROC_results.append(['FPR'] + FPR.tolist())
        ROC_results.append(['TPR'] + TPR.tolist())
        ROC_results.append(['Threshold'] + threshold.tolist())
        ROC_writer.writerows(ROC_results)

        plt.figure()
        plt.title(classes_names[j] + ' ROC CURVE (AUC={:.2f})'.format(AUC))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.plot(FPR, TPR, color='g')
        plt.plot([0, 1], [0, 1], color='m', linestyle='--')
        plt.savefig(ROC_img_path)

        # AP (gt为{0,1})
        AP = average_precision_score(gt, pre_score)
        APs.append(AP)

        # P-R
        PR_csv_path = os.path.join(PR_output, classes_names[j] + '.csv')
        PR_img_path = os.path.join(PR_output, classes_names[j] + '.png')
        PR_f = open(PR_csv_path, 'a', newline='')
        PR_writer = csv.writer(PR_f)
        PR_results = []

        PRECISION, RECALL, thresholds = precision_recall_curve(targets.tolist(), pre_score, pos_label=j)

        PR_results.append(['RECALL'] + RECALL.tolist())
        PR_results.append(['PRECISION'] + PRECISION.tolist())
        PR_results.append(['Threshold'] + thresholds.tolist())
        PR_writer.writerows(PR_results)

        plt.figure()
        plt.title(classes_names[j] + ' P-R CURVE (AP={:.2f})'.format(AP))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.plot(RECALL, PRECISION, color='g')
        plt.savefig(PR_img_path)

    return APs


def main():
    parser = get_parser()
    args = parser.parse_args()

    """
    创建评估文件夹、metrics文件、混淆矩阵文件
    """
    dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = os.path.join('eval_results', args.transfer_loss, dirname)
    metrics_output = os.path.join(save_dir, 'metrics_output.csv')
    prediction_output = os.path.join(save_dir, 'prediction_results.csv')
    os.makedirs(save_dir)

    """
    获取类别名以及对应索引、获取标注文件
    """
    classes_names, indexs = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    """
    测试集并喂入Dataloader
    """
    _, _, target_test_loader, n_class = load_data(args)
    setattr(args, "n_class", n_class)

    setattr(args, "max_iter", args.n_epoch * len(target_test_loader))

    """
    生成模型、加载权重
    """
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model = get_model(args)
    train_state = torch.load(args.resume_from)
    model.load_state_dict(train_state['model_state_dict'])
    model.eval()

    """
    计算Precision、Recall、F1 Score、Confusion matrix
    """
    with torch.no_grad():
        preds, targets, image_paths = [], [], []
        with tqdm(total=len(target_test_loader) // args.batch_size) as pbar:
            for _, batch in enumerate(target_test_loader):
                images, target = batch
                outputs = model.predict(images.to(args.device))
                preds.append(outputs)
                targets.append(target.to(args.device))
                # image_paths.extend(image_path)
                pbar.update(1)
    eval_results = evaluate(torch.cat(preds), torch.cat(targets),
                            ['accuracy', 'precision', 'recall', 'f1_score', 'confusion'], dict(
            topk=(1, 5),
            thrs=None,
            average_mode='none'))

    APs = plot_ROC_curve(torch.cat(preds), torch.cat(targets), classes_names, save_dir)
    get_metrics_output(eval_results, metrics_output, classes_names, indexs, APs)
    # get_prediction_output(torch.cat(preds), torch.cat(targets), image_paths, classes_names, indexs, prediction_output)


if __name__ == "__main__":
    main()

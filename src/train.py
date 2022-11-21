import csv
import nni
import time
import json
import argparse
import warnings
import numpy as np
import torch
import torch.optim as optim

from utils import *
from models import *
from dataloader import *
from train_and_eval import *

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(out_t=None):

    model = Model(param).to(device)
    if param['distill_mode'] == 0:
        optimizer = optim.Adam(model.parameters(), lr=float(1e-2), weight_decay=float(param["weight_decay"]))
    elif param['distill_mode'] == 1:
        optimizer = optim.Adam(model.parameters(), lr=float(param["learning_rate"]), weight_decay=float(param["weight_decay"]))
    criterion_l = torch.nn.NLLLoss()
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    evaluator = get_evaluator(param["dataset"])

    if param['distill_mode'] == 0:
        out, test_acc, test_val, test_best = train_teacher(param, model, g, feats, labels, indices, criterion_l, evaluator, optimizer)
        return out, test_acc, test_val, test_best

        # check_writable(out_t_dir, overwrite=True)
        # np.savez(out_t_dir + "out", out.detach().cpu().numpy())

    else:
        # check_writable(output_dir, overwrite=True)
        # out_t = load_out_t(out_t_dir)
        # out_t = out_t.to(device)

        test_acc, test_val, test_best = train_student(param, model, g, feats, labels, out_t, indices, criterion_l, criterion_t, evaluator, optimizer)
        return test_acc, test_val, test_best
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--dataset", type=str, default="citeseer")
    parser.add_argument("--teacher", type=str, default="GCN", help="Teacher model")
    parser.add_argument("--student", type=str, default="MLP", help="Student model")
    parser.add_argument("--num_heads", type=int, default=4)

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout_t", type=float, default=0.5)
    parser.add_argument("--dropout_s", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.5)
    parser.add_argument("--lamb", type=float,default=0.1)

    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--max_epoch", type=int, default=500)

    parser.add_argument("--distill_mode", type=int, default=0)
    parser.add_argument("--exp_setting", type=str, default="tran", help="[tran, ind]")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--split_rate", type=float, default=0.2)
    parser.add_argument("--save_mode", type=int, default=1)
    parser.add_argument("--data_mode", type=int, default=1)
    parser.add_argument("--ablation_mode", type=int, default=0, help="0: FF-G2M; 1: valinna MLPs; 2: GLNN; 3: LFD; 3: HFD")


    args = parser.parse_args()
    param = args.__dict__
    param.update(nni.get_next_parameter())

    if param['data_mode'] == 0:
        param['dataset'] = 'cora'
    if param['data_mode'] == 1:
        param['dataset'] = 'citeseer'
    if param['data_mode'] == 2:
        param['dataset'] = 'pubmed'
    if param['data_mode'] == 3:
        param['dataset'] = 'coauthor-cs'
    if param['data_mode'] == 4:
        param['dataset'] = 'coauthor-phy'
    if param['data_mode'] == 5:
        param['dataset'] = 'amazon-photo'

    if os.path.exists("../param/best_parameters.json"):
        param = json.loads(open("../param/best_parameters.json", 'r').read())[param['dataset']][param['teacher']]
    param['ablation_mode'] = args.ablation_mode

    g, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
    if args.exp_setting == "tran":
        indices = (idx_train, idx_val, idx_test)
    elif args.exp_setting == "ind":
        indices = graph_split(idx_train, idx_val, idx_test, args.split_rate, args.seed)

    feats = g.ndata["feat"].to(device)
    labels = labels.to(device)
    param['feat_dim'] = g.ndata["feat"].shape[1]
    param['label_dim'] = labels.int().max().item() + 1


    if param['save_mode'] == 0:
        set_seed(param['seed'])
        param["distill_mode"] = 0
        out, _, test_teacher, _ = main()
        param["distill_mode"] = 1
        test_acc, test_val, test_best = main(out)
        nni.report_final_result(test_val)

    else:
        test_acc_list = []
        test_val_list = []
        test_best_list = []
        test_teacher_list = []

        for seed in range(5):
            set_seed(seed + param['seed'])

            if param['data_mode'] == 3 or param['data_mode'] == 4 or param['data_mode'] == 5:
                if param['data_mode'] == 3 or param['data_mode'] == 4:
                    _, _, idx_train, idx_val, idx_test = load_data(args.dataset)
                if args.exp_setting == "tran":
                    indices = (idx_train, idx_val, idx_test)
                elif args.exp_setting == "ind":
                    indices = graph_split(idx_train, idx_val, idx_test, args.split_rate, seed + param['seed'])

            param["distill_mode"] = 0
            out, _, test_teacher, _ = main()
            param["distill_mode"] = 1
            test_acc, test_val, test_best = main(out)
            
            test_acc_list.append(test_acc)
            test_val_list.append(test_val)
            test_best_list.append(test_best)
            test_teacher_list.append(test_teacher)
            nni.report_intermediate_result(test_val)

        nni.report_final_result(np.mean(test_val_list))

    outFile = open('../PerformMetrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    for v, k in param.items():
        results.append(k)
    
    if param['save_mode'] == 0:
        results.append(str(test_acc))
        results.append(str(test_val))
        results.append(str(test_best))
        results.append(str(test_teacher))

    else:  
        results.append(str(test_acc_list))
        results.append(str(test_val_list))
        results.append(str(test_best_list))
        results.append(str(test_teacher_list))
        results.append(str(np.mean(test_acc_list)))
        results.append(str(np.mean(test_val_list)))
        results.append(str(np.mean(test_best_list)))
        results.append(str(np.mean(test_teacher_list)))
        results.append(str(np.std(test_acc_list)))
        results.append(str(np.std(test_val_list)))
        results.append(str(np.std(test_best_list)))
        results.append(str(np.std(test_teacher_list)))
    writer.writerow(results)


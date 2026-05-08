import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt

import HyperLoad
import Utils
from CreateGraph import SLIC, create_graph
from Model.GSDG import GSDG
from Utils import get_class_detail, normal_data_cube, split_data
from scheduler import load_scheduler

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def _resolve_timing_device(device):
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _sync_for_timing(device):
    timing_device = _resolve_timing_device(device)
    if timing_device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(timing_device)


def start_timer(device):
    _sync_for_timing(device)
    return time.perf_counter()


def stop_timer(start_time, device):
    _sync_for_timing(device)
    return time.perf_counter() - start_time


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ExperimentParams():
    parser = argparse.ArgumentParser(description='GSDG')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--path-config', type=str, default='./Config/config.yaml')
    parser.add_argument('-pc', '--print-config', action='store_true', default=False)
    parser.add_argument('-pdi', '--print-data-info', action='store_true', default=False)
    parser.add_argument('-sr', '--show-results', action='store_true', default=False)
    parser.add_argument('--save-results', action='store_true', default=True)
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--iterations', default=10, type=int, help='Number of iterations for averaging results')
    parser.add_argument('--lama', type=float, default=None, help='Fusion parameter for GSDG')
    parser.add_argument('--dk', type=int, default=None, help='d_k parameter for GSDG')
    parser.add_argument('--topk', type=int, default=None, help='topk parameter for GSDG')
    return parser.parse_args()


def build_model(model_name, args, config, height, width, bands, class_num, Q, A, A_Spa, device):
    if model_name != 'GSDG':
        raise ValueError("This release keeps only the GSDG model. Set model.model_name to 'GSDG'.")

    lama = args.lama if args.lama is not None else config["model"].get("lama", 0.95)
    d_k = args.dk if args.dk is not None else config["model"].get("d_k", 16)
    topk = args.topk if args.topk is not None else config["model"].get("topk", 8)
    return GSDG(height, width, bands, class_num, Q, A, A_Spa, model='normal', lama=lama, d_k=d_k, topk=topk).to(device)


def train_and_evaluate(iteration_idx, args, config, seed):
    seed_everything(seed)
    model_name = config["model"]["model_name"]
    dataset_name = config["data_input"]["dataset_name"]
    samples_type = config["data_split"]["samples_type"]
    train_num = config["data_split"]["train_num"]
    train_ratio = config["data_split"]["train_ratio"]
    superpixel_scale = config["data_split"]["superpixel_scale"]
    path_weight = config["result_output"]["path_weight"]
    path_result = config["result_output"]["path_result"]

    dataset_model_dir = os.path.join(path_result, dataset_name, model_name)
    weight_dir = os.path.join(path_weight, dataset_name, model_name)
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(dataset_model_dir, exist_ok=True)

    loader = HyperLoad.LoadingData()
    data, data_gt, height, width, bands, class_num = loader.Loading(dataset_name)
    data = normal_data_cube(data)

    print(f'\n[Iteration {iteration_idx + 1}] Loaded dataset: {dataset_name}')
    print(f'Data shape: height={height}, width={width}, bands={bands}')

    if args.print_data_info:
        print("Class labels:")
        for i, class_name in enumerate(get_class_detail(dataset_name), start=1):
            print(f"number {i}: {class_name}")
        print(f"\nClass count: {class_num}\n")

    gt_reshape = np.reshape(data_gt, [-1])
    train_index, _, test_index = split_data(
        gt_reshape,
        class_num,
        train_ratio,
        0,
        train_num,
        0,
        samples_type,
    )

    empty_val_index = np.array([], dtype=int)
    train_samples_gt, test_samples_gt, _ = create_graph.get_label(gt_reshape, train_index, empty_val_index, test_index)
    train_label_mask, test_label_mask, _ = create_graph.get_label_mask(
        train_samples_gt,
        test_samples_gt,
        np.zeros_like(train_samples_gt),
        data_gt,
        class_num,
    )

    train_gt = np.reshape(train_samples_gt, (height, width))
    test_gt = np.reshape(test_samples_gt, (height, width))
    train_gt_onehot = create_graph.label_to_one_hot(train_gt, class_num)
    test_gt_onehot = create_graph.label_to_one_hot(test_gt, class_num)

    print(f'[Iteration {iteration_idx + 1}] Loading superpixels...')
    Q, _, A, A_Spa, _, preprocess_time = load_slic(
        data,
        train_gt,
        class_num,
        superpixel_scale,
        dataset_name,
        iteration_idx,
    )

    device = args.device
    Q = torch.from_numpy(Q).to(device)
    A = torch.from_numpy(A).to(device)
    A_Spa = torch.from_numpy(A_Spa).to(device)
    net_input = torch.from_numpy(np.array(data, np.float32)).to(device)

    train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
    test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
    train_gt_onehot = torch.from_numpy(train_gt_onehot.astype(np.float32)).to(device)
    test_gt_onehot = torch.from_numpy(test_gt_onehot.astype(np.float32)).to(device)
    train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)

    print(f'[Iteration {iteration_idx + 1}] Preprocessing completed.')

    model = build_model(model_name, args, config, height, width, bands, class_num, Q, A, A_Spa, device)
    print(f'Using model: {model_name}')

    if iteration_idx == 0:
        total_params = count_parameters(model)
        param_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / 1024 ** 2
        print(f"\nModel: {model.__class__.__name__}")
        print(f"Total trainable parameters: {total_params:,}")
        print(f"Parameter size (MiB): {param_size:.2f}")

    optimizer, _, epochs = load_scheduler(model_name, model)
    zeros = torch.zeros([height * width], device=device).float()
    best_metric = float('inf')

    print(f"\n[Iteration {iteration_idx + 1}] Starting training.\n")
    model.train()
    train_start = start_timer(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(net_input)
        loss = Utils.compute_loss(logits, train_gt_onehot, train_label_mask)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            logits = model(net_input)
            train_loss = Utils.compute_loss(logits, train_gt_onehot, train_label_mask)
            train_acc = Utils.compute_acc(logits, train_samples_gt, train_gt_onehot, zeros)

            if train_loss.item() < best_metric:
                best_metric = train_loss.item()
                torch.save(model.state_dict(), os.path.join(weight_dir, f'BestModel_iter{iteration_idx}.pth'))

        torch.cuda.empty_cache()
        model.train()

        if (epoch + 1) % 10 == 0:
            print(f"[Iteration {iteration_idx + 1}/{args.iterations}] Epoch {epoch + 1:4d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train OA: {train_acc * 100:.2f}%")

    train_stage_time = stop_timer(train_start, device)
    training_time = preprocess_time + train_stage_time
    print(f'\n[Iteration {iteration_idx + 1}] Training completed in {training_time:.2f}s.')

    print(f"\n[Iteration {iteration_idx + 1}] Starting testing.\n")
    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(weight_dir, f'BestModel_iter{iteration_idx}.pth')))
        model.eval()
        test_start = start_timer(device)
        logits = model(net_input)
        test_stage_time = stop_timer(test_start, device)
        testing_time = preprocess_time + test_stage_time

        iter_result_path = os.path.join(dataset_model_dir, f'Iteration_{iteration_idx + 1}')
        os.makedirs(iter_result_path, exist_ok=True)

        metrics = Utils.evaluate_performance(logits, test_samples_gt, test_gt_onehot, save_path=iter_result_path)
        print(f'\n[Iteration {iteration_idx + 1}] Testing completed in {testing_time:.2f}s.')
        print(f'[Iteration {iteration_idx + 1}] OA: {metrics["OA"] * 100:.2f}%, '
              f'AA: {metrics["AA"] * 100:.2f}%, Kappa: {metrics["Kappa"] * 100:.2f}%')

    torch.cuda.empty_cache()

    model.eval()
    predicts = model(net_input).detach().cpu().numpy()
    predicts = np.argmax(predicts, axis=1).reshape(height, width) + 1
    Utils.Draw_Classification_Map(
        predicts,
        iter_result_path + f'/predict_map(OA_{metrics["OA"]:.4f})',
        class_count=np.max(predicts),
        dataset=dataset_name,
    )

    return metrics['OA'], metrics['AA'], metrics['Kappa'], metrics['CA'], training_time, testing_time


def load_slic(data, train_gt, class_num, superpixel_scale, dataset_name, iteration_idx):
    slic_loader = SLIC.LDA_SLIC(data, train_gt, class_num - 1)
    start = time.perf_counter()
    save_path = f"{dataset_name}_segmentation_iter{iteration_idx}.png"
    q, s, a, a_spa, seg = slic_loader.simple_superpixel(scale=superpixel_scale, save_path=save_path)
    preprocess_time = time.perf_counter() - start
    print(f"Superpixel preprocessing time: {preprocess_time:.2f}s")
    print(f"Superpixel feature shape: {s.shape}; assignment matrix shape: {q.shape}")
    return q, s, a, a_spa, seg, preprocess_time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def record_output(seed_list, dataset_name, superpixel_scale, oa_ae, aa_ae, kappa_ae, element_acc_ae, training_time_ae,
                  testing_time_ae, save_path):
    oa_mean, oa_std = np.mean(oa_ae), np.std(oa_ae)
    aa_mean, aa_std = np.mean(aa_ae), np.std(aa_ae)
    kappa_mean, kappa_std = np.mean(kappa_ae), np.std(kappa_ae)
    element_acc_mean, element_acc_std = np.mean(element_acc_ae, axis=0), np.std(element_acc_ae, axis=0)
    training_time_mean, training_time_std = np.mean(training_time_ae), np.std(training_time_ae)
    testing_time_mean, testing_time_std = np.mean(testing_time_ae), np.std(testing_time_ae)

    with open(save_path, 'w') as f:
        f.write(f'dataset_name: {dataset_name}\n')
        f.write(f'superpixel_scale: {superpixel_scale}\n')
        f.write(f'seed_list: {seed_list}\n')
        f.write('====================================================================================\n')
        f.write(f'OA: {oa_mean * 100:.2f}+/-{oa_std * 100:.2f}%\n')
        f.write(f'AA: {aa_mean * 100:.2f}+/-{aa_std * 100:.2f}%\n')
        f.write(f'Kappa: {kappa_mean * 100:.2f}+/-{kappa_std * 100:.2f}%\n')
        f.write(f'Training time: {training_time_mean:.2f}+/-{training_time_std:.2f}s\n')
        f.write(f'Testing time: {testing_time_mean:.2f}+/-{testing_time_std:.2f}s\n')
        f.write('-------------------------------------------------------------------------------------\n')
        f.write('Class-specific accuracy:\n')
        for i, (mean_acc, std_acc) in enumerate(zip(element_acc_mean, element_acc_std)):
            f.write(f'Class {i + 1}: {mean_acc * 100:.2f}+/-{std_acc * 100:.2f}%\n')
        f.write('====================================================================================\n')

    print('\n====================================================================================')
    print(f'OA: {oa_mean * 100:.2f}+/-{oa_std * 100:.2f}%')
    print(f'AA: {aa_mean * 100:.2f}+/-{aa_std * 100:.2f}%')
    print(f'Kappa: {kappa_mean * 100:.2f}+/-{kappa_std * 100:.2f}%')
    print(f'Training time: {training_time_mean:.2f}+/-{training_time_std:.2f}s')
    print(f'Testing time: {testing_time_mean:.2f}+/-{testing_time_std:.2f}s')
    print('-------------------------------------------------------------------------------------')
    print('Class-specific accuracy:')
    for i, (mean_acc, std_acc) in enumerate(zip(element_acc_mean, element_acc_std)):
        print(f'Class {i + 1}: {mean_acc * 100:.2f}+/-{std_acc * 100:.2f}%')
    print('====================================================================================')

    return oa_mean, aa_mean, kappa_mean, element_acc_mean, training_time_mean, testing_time_mean


if __name__ == '__main__':
    args = ExperimentParams()

    config = yaml.load(open(args.path_config, "r"), Loader=yaml.FullLoader)
    if args.print_config:
        print(config)

    dataset_name = config["data_input"]["dataset_name"]
    superpixel_scale = config["data_split"]["superpixel_scale"]
    model_name = config["model"]["model_name"]
    path_result = config["result_output"]["path_result"]
    dataset_model_dir = os.path.join(path_result, dataset_name, model_name)
    os.makedirs(dataset_model_dir, exist_ok=True)

    seed_list = [2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034]
    iterations = args.iterations
    oa_ae, aa_ae, kappa_ae, element_acc_ae, training_time_ae, testing_time_ae = [], [], [], [], [], []

    print(f"Starting {iterations} iterations on dataset: {dataset_name} with model: {model_name}.")
    print(f"Using seeds: {seed_list[:iterations]}")

    for i in range(iterations):
        torch.cuda.empty_cache()
        current_seed = seed_list[i % len(seed_list)]
        print(f"\n\n======== Iteration {i + 1}/{iterations} | dataset={dataset_name} | "
              f"model={model_name} | seed={current_seed} ========")
        oa, aa, kappa, element_acc, train_time, test_time = train_and_evaluate(i, args, config, current_seed)
        oa_ae.append(oa)
        aa_ae.append(aa)
        kappa_ae.append(kappa)
        element_acc_ae.append(element_acc)
        training_time_ae.append(train_time)
        testing_time_ae.append(test_time)

    summary_path = os.path.join(
        dataset_model_dir,
        f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{dataset_name}.txt',
    )
    record_output(seed_list, dataset_name, superpixel_scale, oa_ae, aa_ae, kappa_ae, element_acc_ae, training_time_ae,
                  testing_time_ae, summary_path)

    print(f"\nAll iterations completed. Summary saved to {summary_path}")

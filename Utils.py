import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import cohen_kappa_score

colors_indian = [
    '#000000',  # 0 background
    '#FF1E0D',
    '#43FF05',
    '#1600FF',
    '#F1FF3C',
    '#7DFFFF',
    '#FF30FF',
    '#AD3365',
    '#46965C',
    '#BC24FF',
    '#F17A51',
    '#9FFFE3',
    '#D873EA',
    '#A05331',
    '#8EFF1B',
    '#D5BEDB',
    '#F61E0E',
]
cmap_indian = ListedColormap(colors_indian)

colors_pavia = [
    '#000000',
    '#D8BFD8',
    '#00FF00',
    '#00FFFF',
    '#2D8A56',
    '#FF00FF',
    '#FFA500',
    '#9F1FEF',
    '#FF0000',
    '#FFFF00',
]

cmap_pavia = ListedColormap(colors_pavia)

colors_sali = [
    '#000000',
    '#313E9B',
    '#3C53A5',
    '#435CAA',
    '#4C7CBE',
    '#62C4E9',
    '#86DBDA',
    '#87D7AD',
    '#9BD87B',
    '#BBE354',
    '#E2F54E',
    '#E5BD38',
    '#E97E2E',
    '#EE4827',
    '#F22A26',
    '#B62829',
    '#7E1719',
]
cmap_sali = ListedColormap(colors_sali)

colors_HC = [
    '#000000',
    '#8C432E',
    '#0000FF',
    '#FF6400',
    '#00FFC8',
    '#A44B9B',
    '#65AEFF',
]

cmap_HC = ListedColormap(colors_HC)

colors_hanchuan = [
    '#000000',
    '#B03060',
    '#00FFFF',
    '#FF00FE',
    '#A020EF',
    '#7EFFD4',
    '#80FF00',
    '#00CD00',
    '#00FF01',
    '#018B00',
    '#FE0000',
    '#D7BFD7',
    '#FF7F50',
    '#A0522C',
    '#FFFFFF',
    '#DA70D5',
    '#0000FE',
]

cmap_hanchuan = ListedColormap(colors_hanchuan)

DATASET_CMAPS = {
    'indian': cmap_indian,
    'sali': cmap_sali,
    'paviau': cmap_pavia,
    'hanchuan': cmap_hanchuan,
    'HC': cmap_HC,
    'NF': cmap_pavia,
    'UP': cmap_pavia,
}


def Draw_Classification_Map(label, name: str, class_count, scale: float = 4.0, dpi: int = 400, dataset='indian'):
    fig, ax = plt.subplots()
    numlabel = np.array(label, dtype=int)
    if dataset not in DATASET_CMAPS:
        dataset = 'indian'
    plt.imshow(numlabel, cmap=DATASET_CMAPS[dataset], interpolation='none', vmin=0, vmax=class_count)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png',
                    transparent=True, dpi=dpi, pad_inches=0)


def normal_data_cube(data_cube):
    return (data_cube - np.min(data_cube)) / (np.max(data_cube) - np.min(data_cube))


def compute_loss(output, train_samples_gt_onehot, train_label_mask):
    real_label = train_samples_gt_onehot
    loss = -torch.mul(real_label, torch.log(output + 1e-12))
    loss = torch.mul(loss, train_label_mask)
    return torch.sum(loss)


def compute_acc(network_output, train_samples_gt, train_samples_gt_onehot, zeros):
    with torch.no_grad():
        available_label_idx = (train_samples_gt != 0).float()
        available_label_count = available_label_idx.sum()
        correct_prediction = torch.where(
            torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
            available_label_idx,
            zeros,
        ).sum()
        return correct_prediction.cpu() / available_label_count


def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, save_path=None):
    with torch.no_grad():
        available_label_idx = (train_samples_gt != 0).float()
        available_label_count = available_label_idx.sum()
        predictions = torch.argmax(network_output, dim=1)
        ground_truth = torch.argmax(train_samples_gt_onehot, dim=1)

        valid_preds = predictions[available_label_idx.bool()]
        valid_labels = ground_truth[available_label_idx.bool()]
        num_classes = network_output.shape[1]

        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        for true_label, pred_label in zip(valid_labels.cpu().numpy(), valid_preds.cpu().numpy()):
            confusion_matrix[true_label, pred_label] += 1

        correct_predictions = np.trace(confusion_matrix)
        OA = correct_predictions / available_label_count.item()

        per_class_accuracy = np.zeros(num_classes, dtype=np.float32)
        for i in range(num_classes):
            class_total = confusion_matrix[i, :].sum()
            if class_total > 0:
                per_class_accuracy[i] = confusion_matrix[i, i] / class_total

        AA = np.mean(per_class_accuracy)
        Kappa = cohen_kappa_score(valid_labels.cpu().numpy(), valid_preds.cpu().numpy())

        metrics = {
            "OA": OA,
            "AA": AA,
            "Kappa": Kappa,
            "CA": per_class_accuracy.tolist(),
        }

        if save_path:
            save_path = os.path.join(save_path, "Classification_Performance.txt")
            with open(save_path, "w") as f:
                f.write("Classification Metrics:\n")
                f.write(f"Overall Accuracy (OA): {OA:.4f}\n")
                f.write(f"Average Accuracy (AA): {AA:.4f}\n")
                f.write(f"Kappa Coefficient: {Kappa:.4f}\n")
                f.write("Per Class Accuracy:\n")
                for i, acc in enumerate(per_class_accuracy, start=1):
                    f.write(f"  Class {i}: {acc:.4f}\n")

        return metrics


def split_data(labels, class_num, train_ratio=0.1, val_ratio=0.1, train_num=None, val_num=None, samples_type='ratio'):
    train_index, val_index, test_index = [], [], []

    for i in range(1, class_num + 1):
        idx = np.where(labels == i)[0]
        samples_count = len(idx)

        if samples_count == 0:
            print(f"Class {i}: No samples found.")
            continue

        np.random.shuffle(idx)

        if samples_type == 'ratio':
            train_size = int(np.ceil(samples_count * train_ratio))
            val_size = int(np.ceil(samples_count * val_ratio))
        elif samples_type == 'fixed':
            train_size = min(train_num, samples_count) if train_num else 0
            val_size = min(val_num, samples_count - train_size) if val_num else 0
        else:
            raise ValueError("Invalid samples_type. Choose 'ratio' or 'fixed'.")

        train_index.append(idx[:train_size])
        val_index.append(idx[train_size:train_size + val_size])
        test_index.append(idx[train_size + val_size:])

        print(f"Class {i}: Total {samples_count}, Train {train_size}, Val {val_size}, "
              f"Test {samples_count - train_size - val_size}")
        if i == class_num:
            print()

    train_index = np.concatenate(train_index, axis=0) if train_index else np.array([], dtype=int)
    val_index = np.concatenate(val_index, axis=0) if val_index else np.array([], dtype=int)
    test_index = np.concatenate(test_index, axis=0) if test_index else np.array([], dtype=int)

    return train_index, val_index, test_index

def get_class_detail(flag):
    """Return class names for a dataset."""
    Map = {
        'tea': ['Massonpine', 'Bambooforeste', 'Tea plante', 'Reede', 'Rice-paddye', 'Sweet potato',
                'Carawaye', 'Weedw', 'Waterbodye', 'Building/roade'],

        'xuzhou': ['Bareland-1', 'Trees', 'Bareland-2', 'Coals', 'Crops-1', 'Cement', 'Crops-2', 'Lakes', 'Red-tiles'],
        'botswana': ['Water', 'Hippo grass', 'Floodplain grasses 1', 'Floodplain grasses 2', 'Reeds', 'Riparian',
                     'Firescar', 'Island interior', 'Acacia woodlands', 'Acacia shrublands', 'Acacia grasslands',
                     'Short mopane', 'Mixed mopane', 'Chalcedony'],

        'loukia': ['Dense urban fabric', 'Mineral extraction site', 'Non-irrigated arable land', 'Fruit trees',
                   'Olive groves', 'Broad-leaved forest', 'Coniferous forest', 'Mixed forest',
                   'Dense sclerophyllous vegetation', 'Sparce sclerophyllous vegetation', 'Sparsely vegetated areas',
                   'Rocks and sand', 'Water', 'Coastal water'],

        'houston': ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees', 'Soil', 'Water', 'Residential',
                    'Commercial', 'Road', 'Highway', 'Railway', 'Parking Lot 1', 'Parking Lot 2', 'Tennis Court',
                    'Running Track'],

        'Houston2018': ['Healthy grass', 'Stressed grass', 'Artificial turf', 'Evergreen trees', 'Deciduous trees',
                        'Bare earth', 'Water', 'Residential buildings', 'Non-residential buildings', 'Roads',
                        'Sidewalks', 'Crosswalks', 'Major thoroughfares', 'Highways', 'Railways', 'Paved parking lots',
                        'Unpaved parking lots', 'Cars', 'Trains', 'Stadium seats'],

        'indian': ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees',
                   'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                   'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers'],

        'sali': ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow', 'Fallow_smooth',
                 'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                 'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                 'Vinyard_untrained', 'Vinyard_vertical_trellis'],

        'paviau': ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                   'Self-Blocking Bricks', 'Shadows'],

        'paviac': ['Water', 'Trees', 'Asphalt', 'Self-Blocking Bricks', 'Bitumen', 'Tiles', 'Shadows', 'Meadows',
                   'Bare Soil'],

        'longkou': ['Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean', 'Narrow-leaf soybean', 'Rice',
                    'Water', 'Roads and houses', 'Mixed weed'],

        'hanchuan': ['Strawberry', 'Cowpea', 'Soybean', 'Sorghum', 'Water spinach', 'Watermelon', 'Greens', 'Trees',
                     'Grass', 'Red roof', 'Gray roof', 'Plastic', 'Bare soil', 'Road', 'Bright object', 'Water'],

        'honghu': ['Red roof', 'Road', 'Bare soil', 'Cotton', 'Cotton firewood', 'Rape', 'Chinese cabbage', 'Pakchoi',
                   'Cabbage', 'Tuber mustard', 'Brassica parachinensis', 'Brassica chinensis',
                   'Small Brassica chinensis', 'Lactuca sativa', 'Celtuce', 'Film covered lettuce', 'Romaine lettuce',
                   'Carrot', 'White radish', 'Garlic sprout', 'Broad bean', 'Tree'],

        'ksc': ['Scrub', 'Willow swamp', 'CP hammock', 'Slash pine', 'Oak/Broadleaf', 'Hardwood', 'Swamp',
                'Graminoid marsh', 'Spartina marsh', 'Cattail marsh', 'Salt marsh', 'Mud flats', 'Water'],

        'SZUR1': ['Ficus concinna', 'Ficus macrophylla', 'Litchi chinensis', 'Dimocarpus longan',
                  'Araucaria cunninghamii', 'Acacia auriculiformis', 'Camphora officinarum', 'Ficus elastica',
                  'Livistona chinensis', 'Leucaena leucocephala', 'Roystonea regia',
                  'Mangifera indica', 'Terminalia arjuna', 'Delonix regia', 'Kigelia africana',
                  'Archontophoenix alexandrae', 'Bombax ceiba'],

        'SZUR2': ['Ficus concinna', 'Ficus macrophylla', 'Litchi chinensis', 'Araucaria cunninghamii',
                  'Acacia auriculiformis', 'Ficus elastica', 'Livistona chinensis', 'Leucaena leucocephala',
                  'Roystonea regia', 'Mangifera indica', 'Terminalia arjuna', 'Delonix regia',
                  'Kigelia africana', 'Ficus virens', 'Archontophoenix alexandrae', 'Swietenia mahagoni',
                  'Plumeria', 'Bauhinia purpurea', 'Dracontomelon duperreanum', 'Melaleuca',
                  'Casuarina equisetifolia'],
        'HC': ['Analcime', 'Plagioclase', 'Prehnite', 'High-Ca Pyroxene', 'Serpentine', 'Margarite'],
        'UP': ['Analcime', 'Bassanite', 'High-Ca Pyroxene', 'Illite/Muscovite', 'Low-Ca Pyroxene', 'Mg-Smectite',
               'Monohydrated sulfate', 'Plagioclase', 'Prehnite'],
        'NF': ['Fe-Olivine', 'Epidote', 'Chlorite', 'Bassanite', 'Illite/Muscovite', 'Mg-Carbonate', 'Plagioclase',
               'Prehnite', 'Serpentine'],
        'chi': ['Water', 'Bare soil (school)', 'Bare soil (park)', 'Bare soil (farmland)', 'Natural plants',
                'Weeds in farmland', 'Forest', 'Grass', 'Rice field (grown)', 'Rice field (first stage)', 'Row crops',
                'Plastic house', 'Manmade (non-dark)', 'Manmade (dark)', 'Manmade (blue)', 'Manmade (red)',
                'Manmade grass', 'Asphalt', 'Paved ground'],
        'dioni': ['Dense Urban Fabric', 'Mineral Extraction Sites', 'Non Irigated Arable Land', 'Fruit Trees',
                  'Olve Groves', 'Broad -leaved Forest', 'Coniferous Forest', 'Mixed Forest',
                  'Dense Sderophylous Vegetaton', 'Sparce Sderophylous Vegetation', 'Sparcely Vegetated Areas',
                  'Rocks and Sand', 'Water', 'Coastal Water'],
    }
    return Map[flag]

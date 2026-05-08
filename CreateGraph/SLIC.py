import numpy as np
from skimage.segmentation import slic
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def SegmentsLabelProcess(labels):
    """Re-index superpixel labels so they are contiguous from 0."""
    labels = np.array(labels, np.int64)
    unique_labels = np.unique(labels)
    remap = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    return np.vectorize(remap.get)(labels).astype(np.int64)


def _superpixel_centroids(segments: np.ndarray) -> np.ndarray:
    """Return normalized superpixel centroids as [y, x] coordinates in [0, 1]."""
    height, width = segments.shape
    superpixel_count = np.unique(segments).size
    ys, xs = np.indices((height, width))
    ids = segments.ravel()
    counts = np.bincount(ids, minlength=superpixel_count).astype(np.float64) + 1e-12
    cy = np.bincount(ids, weights=ys.ravel(), minlength=superpixel_count) / counts
    cx = np.bincount(ids, weights=xs.ravel(), minlength=superpixel_count) / counts
    return np.stack(
        [cy / max(height - 1, 1), cx / max(width - 1, 1)],
        axis=1,
    ).astype(np.float32)


def get_A_Spa(segments, k=15, sigma_spa: float | None = None, row_normalize: bool = True) -> np.ndarray:
    """Build a spatial adjacency matrix from superpixel centroids."""
    superpixel_count = np.unique(segments).size
    centroids = _superpixel_centroids(segments)

    d_y = centroids[:, None, 0] - centroids[None, :, 0]
    d_x = centroids[:, None, 1] - centroids[None, :, 1]
    distances = np.sqrt(d_y ** 2 + d_x ** 2, dtype=np.float32) + 1e-12

    if sigma_spa is None:
        nonzero_distances = distances[distances > 0]
        sigma_spa = np.median(nonzero_distances) if nonzero_distances.size > 0 else 1.0

    A_full = np.exp(-(distances ** 2) / (2 * (sigma_spa ** 2))).astype(np.float32)
    np.fill_diagonal(A_full, 0.0)

    k_eff = max(1, min(k, superpixel_count - 1))
    idx = np.argpartition(-A_full, kth=k_eff, axis=1)[:, :k_eff]
    rows = np.arange(superpixel_count)[:, None]
    A_spa = np.zeros_like(A_full, dtype=np.float32)
    A_spa[rows, idx] = A_full[rows, idx]
    A_spa = np.maximum(A_spa, A_spa.T)

    if row_normalize:
        A_spa = A_spa / (A_spa.sum(axis=1, keepdims=True) + 1e-6)
        np.fill_diagonal(A_spa, 0.0)

    return A_spa.astype(np.float32)


class SLIC(object):
    def __init__(self, HSI, labels, n_segments=1000, compactness=3,
                 max_iter=20, sigma=0., min_size_factor=0.3, max_size_factor=2, data=None):
        self.Q = None
        self.S = None
        self.superpixel_count = None
        self.segments = None
        self.n_segments = n_segments
        self.compactness = compactness
        self.data_ = data
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma

        height, width, bands = HSI.shape
        data = np.reshape(HSI, [height * width, bands])
        data = preprocessing.StandardScaler().fit_transform(data)

        self.data = np.reshape(data, [height, width, bands])
        self.labels = labels

    def get_Q_and_S_and_Segments(self, save_path=''):
        img = self.data
        height, width, bands = img.shape
        segments = slic(
            img,
            n_segments=self.n_segments,
            compactness=self.compactness,
            convert2lab=False,
            sigma=self.sigma,
            enforce_connectivity=True,
            min_size_factor=self.min_size_factor,
            max_size_factor=self.max_size_factor,
            slic_zero=False,
            start_label=0,
        )

        if segments.max() + 1 != np.unique(segments).size:
            segments = SegmentsLabelProcess(segments)

        self.segments = segments
        self.superpixel_count = segments.max() + 1
        flat_segments = np.reshape(segments, [-1])
        S = np.zeros([self.superpixel_count, bands], dtype=np.float32)
        Q = np.zeros([width * height, self.superpixel_count], dtype=np.float32)
        x = np.reshape(img, [-1, bands])

        for i in range(self.superpixel_count):
            idx = np.where(flat_segments == i)[0]
            pixels = x[idx]
            S[i] = np.sum(pixels, 0) / len(idx)
            Q[idx, i] = 1

        self.S = S
        self.Q = Q
        return Q, S, self.segments

    def get_A(self, sigma: float):
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        height, width = self.segments.shape

        for i in range(height - 2):
            for j in range(width - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max == sub_min or A[sub_max, sub_min] != 0:
                    continue

                pix1 = self.S[sub_max]
                pix2 = self.S[sub_min]
                distance = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                A[sub_max, sub_min] = A[sub_min, sub_max] = distance

        A_spa = get_A_Spa(self.segments)
        return A, A_spa


class LDA_SLIC(object):
    def __init__(self, data, labels, n_component):
        self.data = data
        self.init_labels = labels
        self.curr_data = data
        self.n_component = n_component
        self.height, self.width, self.bands = data.shape
        self.x_flatt = np.reshape(data, [self.width * self.height, self.bands])
        self.y_flatt = np.reshape(labels, [self.height * self.width])
        self.labes = labels

    def LDA_Process(self, curr_labels):
        curr_labels = np.reshape(curr_labels, [-1])
        labeled_idx = np.where(curr_labels != 0)[0]
        x = self.x_flatt[labeled_idx]
        y = curr_labels[labeled_idx]

        unique_classes = np.unique(y)
        min_samples_per_class = 2
        sufficient_samples = True

        for cls in unique_classes:
            cls_count = np.sum(y == cls)
            if cls_count < min_samples_per_class:
                sufficient_samples = False
                print(f"Warning: class {cls} has only {cls_count} samples; falling back to PCA.")
                break

        if not sufficient_samples or len(unique_classes) < 2:
            from sklearn.decomposition import PCA

            n_components = min(len(unique_classes) - 1, x.shape[1]) if len(unique_classes) > 1 else min(5, x.shape[1])
            n_components = max(1, n_components)
            reducer = PCA(n_components=n_components)
            X_new = reducer.fit_transform(self.x_flatt)
        else:
            reducer = LinearDiscriminantAnalysis()
            reducer.fit(x, y - 1)
            X_new = reducer.transform(self.x_flatt)

        return np.reshape(X_new, [self.height, self.width, -1])

    def SLIC_Process(self, img, scale=50, save_path=None):
        n_segments_init = self.height * self.width / scale
        print("n_segments_init", n_segments_init)
        slic_model = SLIC(
            img,
            n_segments=n_segments_init,
            labels=self.labes,
            compactness=0.1,
            sigma=1,
            min_size_factor=0.1,
            max_size_factor=2,
            data=self.data,
        )
        Q, S, Segments = slic_model.get_Q_and_S_and_Segments(save_path=save_path)
        A, A_Spa = slic_model.get_A(sigma=10)
        return Q, S, A, A_Spa, Segments

    def simple_superpixel(self, scale, save_path=None):
        X = self.LDA_Process(self.init_labels)
        Q, S, A, A_Spa, Seg = self.SLIC_Process(X, scale=scale, save_path=save_path)
        return Q, S, A, A_Spa, Seg

    def simple_superpixel_no_LDA(self, scale, save_path=None):
        Q, S, A, A_Spa, Seg = self.SLIC_Process(self.data, scale=scale, save_path=save_path)
        return Q, S, A, A_Spa, Seg

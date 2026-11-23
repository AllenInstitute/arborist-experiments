import numpy as np
import torch

from arborist.data.datasets import build_tree_sample

from neuron_proofreader.utils.ml_util import TensorDict

from arborist_experiments.tasks.merge_proofreading import (
    GraphMergeDetector,
    MultimodalMergeDetector,
    load_detector,
)
from arborist_experiments.baselines import (
    TMDEncoder,
    persistence_barcode,
    persistence_image,
)


def _y_tree():
    # root -(10 along x)-> node1, then node1 -(5 along x)-> node2 and
    # node1 -(3 along y)-> node3
    curves = [
        np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [5, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [0, 3, 0]], dtype=np.float32),
    ]
    return build_tree_sample(curves, [(0, 1), (1, 2), (1, 3)])


def _assert_bars(bars, expected):
    bars = np.asarray(sorted(map(tuple, bars)))
    assert np.allclose(bars, np.asarray(sorted(expected)), atol=1e-4)


def test_barcode_radial():
    bars = persistence_barcode(_y_tree(), distance="radial")
    _assert_bars(bars, [(np.sqrt(109), 10.0), (15.0, 0.0)])


def test_barcode_path():
    bars = persistence_barcode(_y_tree(), distance="path")
    _assert_bars(bars, [(13.0, 10.0), (15.0, 0.0)])


def test_barcode_one_bar_per_leaf():
    rng = np.random.default_rng(0)
    curves, edges = [], []
    for k in range(1, 30):
        parent = int(rng.integers(0, k))
        steps = rng.normal(size=(4, 3)).astype(np.float32)
        steps[0] = 0
        curves.append(steps)
        edges.append((parent, k))
    sample = build_tree_sample(curves, edges)
    n_leaves = sum(1 for k in range(1, 30) if k not in {p for p, _ in edges})
    assert persistence_barcode(sample, distance="radial").shape == (n_leaves, 2)
    # Path distance grows monotonically from the root, so every branch is
    # born farther out than it dies; radial distance need not be monotone.
    bars = persistence_barcode(sample, distance="path")
    assert bars.shape == (n_leaves, 2)
    assert np.all(bars[:, 0] >= bars[:, 1])
    assert bars[:, 1].min() == 0.0


def test_empty_sample():
    sample = build_tree_sample([], [])
    assert persistence_barcode(sample).shape == (0, 2)
    assert persistence_image(np.zeros((0, 2)), 100.0).sum() == 0


def test_persistence_image():
    img = persistence_image(persistence_barcode(_y_tree()), max_dist=20.0, resolution=10)
    assert img.shape == (10, 10)
    assert img.max() == 1.0
    assert img.min() >= 0.0


def test_tmd_encoder():
    encoder = TMDEncoder(max_dist=20.0, resolution=10, embed_dim=32).eval()
    with torch.no_grad():
        z = encoder([_y_tree(), build_tree_sample([], [])])
    assert z.shape == (2, 32)


def test_graph_detector_forward_and_load(tmp_path):
    model = GraphMergeDetector(TMDEncoder(max_dist=20.0, resolution=10)).eval()
    samples = [_y_tree(), build_tree_sample([], [])]
    with torch.no_grad():
        logits = model(samples, "merge")
        assert logits.shape == (2, 1)
        assert torch.equal(logits, model({"tree_sample": samples}))

    ckpt = tmp_path / "model.pth"
    model.save(ckpt)
    loaded = load_detector(str(ckpt)).eval()
    assert isinstance(loaded, GraphMergeDetector)
    assert loaded.config["model_type"] == "TMDEncoder_graph"
    with torch.no_grad():
        assert torch.equal(loaded(samples), logits)


def test_multimodal_detector_forward_and_load(tmp_path):
    shape = (2, 32, 32, 32)
    model = MultimodalMergeDetector(
        TMDEncoder(max_dist=20.0, resolution=10),
        shape,
        img_embed_dim=16,
        base_channels=4,
        max_channels=8,
        depth=4,
    ).eval()
    samples = [_y_tree(), build_tree_sample([], [])]
    x = TensorDict({"img": torch.rand((2,) + shape), "tree_sample": samples})
    with torch.no_grad():
        logits = model(x, "merge")
    assert logits.shape == (2, 1)

    ckpt = tmp_path / "model.pth"
    model.save(ckpt)
    loaded = load_detector(str(ckpt)).eval()
    assert isinstance(loaded, MultimodalMergeDetector)
    assert loaded.config == model.config
    with torch.no_grad():
        assert torch.equal(loaded(x), logits)

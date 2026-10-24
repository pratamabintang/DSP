"""
Episodic 1-way K-shot dataset for binary landslide segmentation (RGB PNG patches).

Layout (one directory per split, regions are already separated by the split):

    <base_dir>/<split>/image/<name>.png     RGB
    <base_dir>/<split>/label/<name>.png     binary mask (0 = background, >0 = landslide)
    <base_dir>/black_list.txt               optional, one sample stem per line

Every item is one episode in the format consumed by training.py / validation.py
(identical to SuperpixelDataset in GenericSuperDatasetv2.py). After the default
collate with batch size B:

    support_images : [[ B x 3 x H x W ] * n_shots]      (way x shot)  float32, ImageNet-normalised
    support_mask   : [[ {'fg_mask': B x H x W,
                         'bg_mask': B x H x W} ] * n_shots]          float32 in {0, 1}
    query_images   : [ B x 3 x H x W ]                                float32
    query_labels   : [ B x H x W ]                                    float32 in {0, 1} (cast to long by the caller)
    class_ids      : [[ B ]]                                          always 1 (landslide)

Different patches of the same physical landslide may appear as support and/or query in
different episodes. This is intentional and is not filtered.

Train mode  : episodes are drawn at random (query and support from this split), with
              synchronised flip/rot90 plus a mild photometric jitter on the images.
Val/test    : episodes are deterministic (function of `seed` and the index), no augmentation.
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

FG_LABEL = 1
BG_LABEL = 0


def load_blacklist(path: Optional[str]) -> set:
    """Read sample stems to skip. Supports '#' comments and entries with an extension."""
    if not path or not os.path.isfile(path):
        return set()
    out = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if line:
                out.add(os.path.splitext(os.path.basename(line))[0])
    return out


def _find_subdir(parent: str, name: str) -> str:
    """Case-insensitive sub-directory lookup (the layout may use 'image' or 'IMAGE')."""
    for d in os.listdir(parent):
        if d.lower() == name and os.path.isdir(os.path.join(parent, d)):
            return os.path.join(parent, d)
    raise FileNotFoundError(f"'{name}' directory not found in {parent}")


def _list_pngs(directory: str) -> Dict[str, str]:
    return {os.path.splitext(f)[0]: os.path.join(directory, f)
            for f in sorted(os.listdir(directory)) if f.lower().endswith('.png')}


def read_rgb(path: str) -> np.ndarray:
    """H x W x 3 uint8. Raises on unreadable / non-RGB files."""
    with Image.open(path) as im:
        if im.mode not in ('RGB', 'RGBA'):
            raise ValueError(f'expected an RGB image, got mode {im.mode}: {path}')
        return np.array(im.convert('RGB'), dtype=np.uint8)


def read_binary_mask(path: str) -> np.ndarray:
    """H x W uint8 in {0, 1}; any non-zero pixel is landslide."""
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3:  # RGB(A) / LA masks: foreground if any colour channel is set
        arr = arr[..., :3].max(axis=-1) if arr.shape[-1] >= 3 else arr[..., 0]
    return (arr > 0).astype(np.uint8)


class LandslideDataset(Dataset):
    """
    Args:
        base_dir:        dataset root containing train/ val/ test/ (optional if `split_dir` is given)
        split_dir:       explicit directory of this split (containing image/ and label/); overrides
                         <base_dir>/<split>, so the splits may live in unrelated locations
        split:           'train' | 'val' | 'test'
        input_size:      patches are resized to (input_size, input_size). Must stay coupled to
                         the model's feature_hw (backbone stride is 8: 256 -> 32x32).
        n_shots:         support images per episode (the model currently supports 1)
        blacklist_path:  file with corrupted sample stems. 'default' (or empty/None-like config
                         value) -> <base_dir>/black_list.txt if it exists; 'none' -> disabled;
                         any other value is used as the path and must exist.
                         The image/label folder names are matched case-insensitively
                         ('image', 'IMAGE', 'Label', ...), file extensions too ('.png', '.PNG').
        min_fg_pixels:   minimum landslide pixels (native resolution) for a patch to be usable
                         as support (and as query when `query_require_fg`)
        query_require_fg: only sample queries that contain landslide. Default: True in train,
                         False in val/test (empty queries are part of the real distribution).
        length:          episodes per epoch. Default: number of query candidates.
        seed:            seed for the deterministic val/test episodes
        verify:          read every mask once at start-up (validity, size match, fg count)
    """

    def __init__(self, base_dir: Optional[str] = None, split: str = 'train', input_size: int = 256,
                 n_shots: int = 1, blacklist_path: Optional[str] = 'default',
                 min_fg_pixels: int = 64, query_require_fg: Optional[bool] = None,
                 length: Optional[int] = None, seed: int = 1234, verify: bool = True,
                 split_dir: Optional[str] = None):
        super().__init__()
        assert split in ('train', 'val', 'test'), f'unknown split {split}'
        self.split = split
        self.is_train = split == 'train'
        self.input_size = int(input_size)
        self.n_shots = int(n_shots)
        self.min_fg_pixels = int(min_fg_pixels)
        self.seed = int(seed)
        self.query_require_fg = self.is_train if query_require_fg is None else bool(query_require_fg)

        if split_dir in (None, '', 'None'):
            if not base_dir:
                raise ValueError('either base_dir or split_dir must be given')
            split_dir = os.path.join(base_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f'split directory not found: {split_dir}')
        if blacklist_path in (None, '', 'default'):
            blacklist_path = os.path.join(base_dir, 'black_list.txt') if base_dir else None
        elif str(blacklist_path).lower() == 'none':
            blacklist_path = None
        elif not os.path.isfile(blacklist_path):
            raise FileNotFoundError(f'blacklist file not found: {blacklist_path}')
        self.blacklist = load_blacklist(blacklist_path)
        print(f'[LandslideDataset:{split}] split_dir={split_dir} blacklist={blacklist_path} '
              f'({len(self.blacklist)} entries)')

        images = _list_pngs(_find_subdir(split_dir, 'image'))
        labels = _list_pngs(_find_subdir(split_dir, 'label'))

        # --- sanity checks: pairing, readability, channels, mask values, emptiness ---
        self.samples: List[Dict] = []   # {'name', 'image_path', 'label_path', 'fg_pixels'}
        self.skipped: List[Tuple[str, str]] = []
        for name, img_path in images.items():
            if name in self.blacklist:
                self.skipped.append((name, 'blacklisted'))
                continue
            if name not in labels:
                self.skipped.append((name, 'no matching label'))
                continue
            sample = {'name': name, 'image_path': img_path, 'label_path': labels[name], 'fg_pixels': -1}
            if verify:
                problem = self._verify(sample)
                if problem:
                    self.skipped.append((name, problem))
                    continue
            self.samples.append(sample)
        for name in labels:
            if name not in images and name not in self.blacklist:
                self.skipped.append((name, 'no matching image'))

        if not self.samples:
            raise RuntimeError(f'no valid samples in {split_dir} (skipped: {len(self.skipped)})')
        if self.skipped:
            reasons = {}
            for _, r in self.skipped:
                reasons[r] = reasons.get(r, 0) + 1
            print(f'[LandslideDataset:{split}] skipped {len(self.skipped)} samples: {reasons}')

        # support patches must contain landslide; the prototype is undefined otherwise
        self.fg_idx = [i for i, s in enumerate(self.samples)
                       if s['fg_pixels'] < 0 or s['fg_pixels'] >= self.min_fg_pixels]
        if not self.fg_idx:
            raise RuntimeError(f'[{split}] no patch has >= {self.min_fg_pixels} landslide pixels')
        self.query_idx = self.fg_idx if self.query_require_fg else list(range(len(self.samples)))
        self.length = int(length) if length else len(self.query_idx)
        print(f'[LandslideDataset:{split}] {len(self.samples)} patches, {len(self.fg_idx)} usable as support, '
              f'{len(self.query_idx)} query candidates, {self.length} episodes/epoch')

    # ------------------------------------------------------------------ checks
    @staticmethod
    def _verify(sample: Dict) -> Optional[str]:
        """Return a problem description or None. Fills sample['fg_pixels']."""
        try:
            with Image.open(sample['image_path']) as im:
                if im.mode not in ('RGB', 'RGBA'):
                    return f'image mode {im.mode} is not RGB'
                img_size = im.size
                im.load()
        except Exception as e:
            return f'unreadable image ({type(e).__name__})'
        try:
            mask = read_binary_mask(sample['label_path'])
            with Image.open(sample['label_path']) as lb:
                raw_size = lb.size
        except Exception as e:
            return f'unreadable label ({type(e).__name__})'
        if raw_size != img_size:
            return f'image/label size mismatch {img_size} vs {raw_size}'
        sample['fg_pixels'] = int(mask.sum())
        return None

    # ------------------------------------------------------------------ loading
    def _load_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        s = self.samples[idx]
        img = read_rgb(s['image_path'])
        lb = read_binary_mask(s['label_path'])
        if img.shape[:2] != lb.shape:
            raise ValueError(f"image/label size mismatch for {s['name']}: {img.shape[:2]} vs {lb.shape}")
        return img, lb

    def _to_tensors(self, img: np.ndarray, lb: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        size = (self.input_size, self.input_size)
        if img.shape[:2] != size:
            img = np.array(Image.fromarray(img).resize(size, Image.BILINEAR))
            lb = np.array(Image.fromarray(lb).resize(size, Image.NEAREST))

        x = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0   # 3 x H x W in [0, 1]
        y = torch.from_numpy(lb.copy()).float()                              # H x W in {0, 1}

        if self.is_train:
            # synchronised geometric augmentation
            if torch.rand(1).item() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1).item() < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
            # photometric jitter (image only)
            if torch.rand(1).item() < 0.5:
                contrast = 0.8 + 0.4 * torch.rand(1).item()
                brightness = (torch.rand(1).item() - 0.5) * 0.2
                x = (x * contrast + brightness).clamp(0, 1)

        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (x - mean) / std, y

    @staticmethod
    def _fg_bg(label: torch.Tensor) -> Dict[str, torch.Tensor]:
        fg = (label == FG_LABEL).float()
        return {'fg_mask': fg, 'bg_mask': 1.0 - fg}

    # ------------------------------------------------------------------ episodes
    def _sample_episode(self, index: int) -> Tuple[int, List[int]]:
        """Pick (query index, support indices) into self.samples."""
        if self.is_train:
            q = self.query_idx[int(torch.randint(len(self.query_idx), (1,)).item())]
            gen = None
        else:
            q = self.query_idx[index % len(self.query_idx)]
            gen = torch.Generator().manual_seed(self.seed * 1000003 + index)

        pool = [i for i in self.fg_idx if i != q]
        if len(pool) < self.n_shots:
            raise RuntimeError(f'[{self.split}] need {self.n_shots} support patches besides the query, '
                               f'only {len(pool)} available')
        order = torch.randperm(len(pool), generator=gen)[:self.n_shots].tolist()
        return q, [pool[i] for i in order]

    def __getitem__(self, index: int) -> Dict:
        q, sups = self._sample_episode(index)

        support_images, support_mask = [], []
        for si in sups:
            x, y = self._to_tensors(*self._load_pair(si))
            support_images.append(x)
            support_mask.append(self._fg_bg(y))

        qx, qy = self._to_tensors(*self._load_pair(q))

        return {
            'class_ids': [[FG_LABEL] * self.n_shots],
            'support_images': [support_images],
            'support_mask': [support_mask],
            'query_images': [qx],
            'query_labels': [qy],
            # metadata
            'support_image_path': self.samples[sups[0]]['image_path'],
            'support_label_path': self.samples[sups[0]]['label_path'],
            'query_image_path': self.samples[q]['image_path'],
            'query_label_path': self.samples[q]['label_path'],
        }

    def __len__(self) -> int:
        return self.length

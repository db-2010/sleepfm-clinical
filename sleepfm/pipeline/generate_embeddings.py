import yaml
import torch
from torch import nn
from loguru import logger
import os
import sys

sys.path.append("../")
from utils import *
from models.dataset import SetTransformerDataset, collate_fn
from models.models import SetTransformer
import click
import time
import math
import datetime
import numpy as np
import tqdm
import shutil
import wandb
import h5py


def _write_root_attrs(
        hdf5_file: h5py.File,
        *,
        sampling_freq: int,
        window_s: float,
        stride_s: float,
        time_offset_s: float = 0.0,
        export_type: str,
        source_file: str,
        patch_size: int,
        embed_dim: int,
) -> None:
    """Write timing/provenance metadata once per output file."""
    hdf5_file.attrs["sampling_freq"] = int(sampling_freq)
    hdf5_file.attrs["window_s"] = float(window_s)
    hdf5_file.attrs["stride_s"] = float(stride_s)
    hdf5_file.attrs["time_offset_s"] = float(time_offset_s)
    hdf5_file.attrs["export_type"] = str(export_type)
    hdf5_file.attrs["source_file"] = str(source_file)
    hdf5_file.attrs["patch_size"] = int(patch_size)
    hdf5_file.attrs["embed_dim"] = int(embed_dim)


def _ensure_window_start_dataset(hdf5_file: h5py.File, required_len: int) -> h5py.Dataset:
    """Create or resize the explicit window start dataset."""
    if "window_start_s" not in hdf5_file:
        dset = hdf5_file.create_dataset(
            "window_start_s",
            shape=(required_len,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=True,
        )
        return dset

    dset = hdf5_file["window_start_s"]
    if dset.shape[0] < required_len:
        dset.resize((required_len,))
    return dset


def _write_window_starts_slice(
        hdf5_file: h5py.File,
        *,
        start_idx: int,
        n_rows: int,
        stride_s: float,
        time_offset_s: float = 0.0,
) -> None:
    """Write explicit per-row start times for a chunk slice."""
    end_idx = start_idx + n_rows
    dset = _ensure_window_start_dataset(hdf5_file, end_idx)
    starts = time_offset_s + (np.arange(start_idx, end_idx, dtype=np.float64) * float(stride_s))
    dset[start_idx:end_idx] = starts


def _ensure_modality_dataset(
        hdf5_file: h5py.File,
        modality_type: str,
        chunk_arr: np.ndarray,
) -> h5py.Dataset:
    """
    Create or return a modality dataset.

    Expected chunk_arr shape:
        (n_rows, embedding_dim)
    """
    if modality_type in hdf5_file:
        return hdf5_file[modality_type]

    return hdf5_file.create_dataset(
        modality_type,
        data=chunk_arr,
        chunks=True,
        maxshape=(None,) + chunk_arr.shape[1:],
    )


def _append_or_write_chunk(
        hdf5_file: h5py.File,
        *,
        modality_type: str,
        chunk_arr: np.ndarray,
        start_idx: int,
) -> None:
    """
    Write a chunk into the target dataset at explicit row offsets.
    """
    dset = _ensure_modality_dataset(hdf5_file, modality_type, chunk_arr)

    end_idx = start_idx + chunk_arr.shape[0]
    if dset.shape[0] < end_idx:
        dset.resize((end_idx,) + dset.shape[1:])

    dset[start_idx:end_idx] = chunk_arr


def _sample_start_to_row_index(
        chunk_start_samples: int,
        *,
        sampling_freq: int,
        stride_s: float,
) -> int:
    """
    Convert a sample-based chunk start into an embedding row index.
    """
    stride_samples = int(round(float(stride_s) * int(sampling_freq)))
    if stride_samples <= 0:
        raise ValueError(f"Invalid stride_samples={stride_samples}")

    if chunk_start_samples % stride_samples != 0:
        logger.warning(
            f"chunk_start_samples={chunk_start_samples} is not divisible by stride_samples={stride_samples}. "
            f"Using floor division; this may indicate an upstream alignment issue."
        )

    return int(chunk_start_samples // stride_samples)


@click.command("generate_embeddings")
@click.option("--model_path", type=str, default='path')
@click.option("--dataset_name", type=str, default='mesa')
@click.option("--channel_groups_path", type=str, default='../configs/channel_groups.json')
@click.option("--split_path", type=str, default='../configs/dataset_split.json')
@click.option("--splits", type=str, default='train,validation,test')
@click.option("--num_workers", type=int, default=16)
@click.option("--batch_size", type=int, default=128)
def generate_embeddings(
    model_path,
    dataset_name,
    channel_groups_path,
    split_path,
    splits,
    num_workers,
    batch_size
):
    config_path = os.path.join(model_path, "config.json")
    config = load_config(config_path)
    channel_groups = load_data(channel_groups_path)

    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_name = dataset_name.lower()

    output = os.path.join(model_path, f"{dataset_name}")
    output_5min_agg = os.path.join(model_path, f"{dataset_name}_5min_agg")
    os.makedirs(output, exist_ok=True)
    os.makedirs(output_5min_agg, exist_ok=True)

    modality_types = config["modality_types"]
    in_channels = config["in_channels"]
    patch_size = config["patch_size"]
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    pooling_head = config["pooling_head"]
    dropout = 0.0

    data_path = config["data_path"]
    sampling_freq = int(config["sampling_freq"])
    sampling_duration_min = float(config["sampling_duration"])

    token_window_s = 5.0
    token_stride_s = 5.0
    agg_window_s = sampling_duration_min * 60.0
    agg_stride_s = sampling_duration_min * 60.0

    expected_patch_size = int(round(token_window_s * sampling_freq))
    if patch_size != expected_patch_size:
        raise ValueError(
            f"Unexpected patch_size={patch_size}. Expected {expected_patch_size} "
            f"from token_window_s={token_window_s} and sampling_freq={sampling_freq}."
        )

    logger.info(f"Output Path: {output}")
    logger.info(f"Output 5 Min Agg Path: {output_5min_agg}")
    logger.info(f"modality_types: {modality_types}")
    logger.info(f"sampling_freq: {sampling_freq}")
    logger.info(f"token_window_s/stride_s: {token_window_s}/{token_stride_s}")
    logger.info(f"agg_window_s/stride_s: {agg_window_s}/{agg_stride_s}")
    logger.info(f"Batch Size: {batch_size}; Number of Workers: {num_workers}")

    device = torch.device("cuda")
    logger.info(f"Device set to Cuda")

    start = time.time()
    split_dataset = load_data(split_path)
    splits = splits.split(",")

    if dataset_name.lower() in ["shhs1", "shhs2"]:
        path_to_data = os.path.join(data_path, f"SHHS/{dataset_name}")
        hdf5_paths = [os.path.join(path_to_data, file_name) for file_name in os.listdir(path_to_data)]
    else:
        hdf5_paths = []
        for split in splits:
            filtered_files = [fp for fp in split_dataset[split] if dataset_name in fp.lower()]
            hdf5_paths += filtered_files

        hdf5_paths = [os.path.join(data_path, file) for file in hdf5_paths]

    logger.info(f"Number of files to process: {len(hdf5_paths)}")

    dataset = SetTransformerDataset(config, channel_groups, hdf5_paths=hdf5_paths, split="test")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn
    )

    logger.info(f"Dataset loaded in {time.time() - start:.1f} seconds")

    model_class = getattr(sys.modules[__name__], config['model'])
    logger.info(f"Model Class: {config['model']}")
    model = model_class(
        in_channels,
        patch_size,
        embed_dim,
        num_heads,
        num_layers,
        pooling_head=pooling_head,
        dropout=dropout
    )
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)
    total_layers, total_params = count_parameters(model)
    logger.info(f'Trainable parameters: {total_params / 1e6:.2f} million')
    logger.info(f'Number of layers: {total_layers}')

    checkpoint = torch.load(os.path.join(model_path, "best.pt"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for batch in dataloader:
                batch_data, mask_list, file_paths, dset_names_list, chunk_starts = batch
                (bas, resp, ekg, emg) = batch_data
                (mask_bas, mask_resp, mask_ekg, mask_emg) = mask_list

                bas = bas.to(device, dtype=torch.float)
                resp = resp.to(device, dtype=torch.float)
                ekg = ekg.to(device, dtype=torch.float)
                emg = emg.to(device, dtype=torch.float)

                mask_bas = mask_bas.to(device, dtype=torch.bool)
                mask_resp = mask_resp.to(device, dtype=torch.bool)
                mask_ekg = mask_ekg.to(device, dtype=torch.bool)
                mask_emg = mask_emg.to(device, dtype=torch.bool)

                embeddings = [
                    model(bas, mask_bas),
                    model(resp, mask_resp),
                    model(ekg, mask_ekg),
                    model(emg, mask_emg),
                ]

                # ------------------------------------------------------------
                # 5 min aggregated embeddings
                # ------------------------------------------------------------
                embeddings_5min = [e[0].unsqueeze(1) for e in embeddings]

                for i in range(len(file_paths)):
                    file_path = file_paths[i]
                    chunk_start_samples = int(chunk_starts[i])
                    subject_id = os.path.basename(file_path).split('.')[0]
                    output_path = os.path.join(output_5min_agg, f"{subject_id}.hdf5")

                    chunk_start_idx = _sample_start_to_row_index(
                        chunk_start_samples,
                        sampling_freq=sampling_freq,
                        stride_s=agg_stride_s,
                    )

                    with h5py.File(output_path, 'a') as hdf5_file:
                        _write_root_attrs(
                            hdf5_file,
                            sampling_freq=sampling_freq,
                            window_s=agg_window_s,
                            stride_s=agg_stride_s,
                            time_offset_s=0.0,
                            export_type="5min_agg",
                            source_file=file_path,
                            patch_size=patch_size,
                            embed_dim=embed_dim,
                        )

                        n_rows_written = None
                        for modality_idx, modality_type in enumerate(config["modality_types"]):
                            chunk_arr = embeddings_5min[modality_idx][i].cpu().numpy()
                            _append_or_write_chunk(
                                hdf5_file,
                                modality_type=modality_type,
                                chunk_arr=chunk_arr,
                                start_idx=chunk_start_idx,
                            )
                            if n_rows_written is None:
                                n_rows_written = chunk_arr.shape[0]

                        if n_rows_written is not None:
                            _write_window_starts_slice(
                                hdf5_file,
                                start_idx=chunk_start_idx,
                                n_rows=n_rows_written,
                                stride_s=agg_stride_s,
                                time_offset_s=0.0,
                            )

                # ------------------------------------------------------------
                # 5 s token embeddings
                # ------------------------------------------------------------
                embeddings_5s = [e[1] for e in embeddings]

                for i in range(len(file_paths)):
                    file_path = file_paths[i]
                    chunk_start_samples = int(chunk_starts[i])
                    subject_id = os.path.basename(file_path).split('.')[0]
                    output_path = os.path.join(output, f"{subject_id}.hdf5")

                    chunk_start_idx = _sample_start_to_row_index(
                        chunk_start_samples,
                        sampling_freq=sampling_freq,
                        stride_s=token_stride_s,
                    )

                    with h5py.File(output_path, 'a') as hdf5_file:
                        _write_root_attrs(
                            hdf5_file,
                            sampling_freq=sampling_freq,
                            window_s=token_window_s,
                            stride_s=token_stride_s,
                            time_offset_s=0.0,
                            export_type="5s",
                            source_file=file_path,
                            patch_size=patch_size,
                            embed_dim=embed_dim,
                        )

                        n_rows_written = None
                        for modality_idx, modality_type in enumerate(config["modality_types"]):
                            chunk_arr = embeddings_5s[modality_idx][i].cpu().numpy()
                            _append_or_write_chunk(
                                hdf5_file,
                                modality_type=modality_type,
                                chunk_arr=chunk_arr,
                                start_idx=chunk_start_idx,
                            )
                            if n_rows_written is None:
                                n_rows_written = chunk_arr.shape[0]

                        if n_rows_written is not None:
                            _write_window_starts_slice(
                                hdf5_file,
                                start_idx=chunk_start_idx,
                                n_rows=n_rows_written,
                                stride_s=token_stride_s,
                                time_offset_s=0.0,
                            )

                pbar.update()


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


if __name__ == '__main__':
    generate_embeddings()
#!/usr/bin/env python3
"""
Data preprocessing pipeline for multimodal compound–protein interaction prediction.

This script prepares DrugBAN-style compound–protein interaction datasets for the
GraphBAN-Multimodal-CPI framework. It generates residue-level ESMC embeddings,
extracts Foldseek/3Di structural tokens from predicted protein structures, generates
SaProt structure-aware embeddings, and saves processed train/validation/test
partitions as pickle files.

Expected input files
--------------------
For the DrugBAN clustered split:
    source_train.csv
    target_train.csv
    target_test.csv

For numbered seed splits:
    source_train_<dataset><seed>.csv
    target_train_<dataset><seed>.csv
    target_test_<dataset><seed>.csv

Each input file is expected to contain at least:
    - uniprot_id
    - SMILES
    - Y

Example
-------
python data_processing.py \
    --dataset bindingdb \
    --seed drugban \
    --input-dir ./data/bindingdb/drugban \
    --cif-dir ./data/bindingdb/CIF \
    --output-dir ./outputs/bindingdb/drugban
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from transformers import AutoModel, AutoTokenizer


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_foldseek(foldseek_arg: str | None = None) -> str:
    """Return an executable Foldseek path."""
    foldseek_bin = foldseek_arg or os.environ.get("FOLDSEEK") or shutil.which("foldseek")
    if not foldseek_bin:
        raise RuntimeError(
            "Foldseek was not found. Set FOLDSEEK=/path/to/foldseek, "
            "add Foldseek to PATH, or pass --foldseek."
        )

    foldseek_path = Path(foldseek_bin)
    if not (foldseek_path.is_file() and os.access(foldseek_path, os.X_OK)):
        raise RuntimeError(f"Foldseek is not executable: {foldseek_path}")

    return str(foldseek_path)


def get_protein_features_token_level(
    sequences: Iterable[str],
    model: ESMC,
    max_len: int = 1024,
    feature_dim: int | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """
    Generate token-level ESMC embeddings for protein sequences.

    Parameters
    ----------
    sequences
        Protein sequences.
    model
        Loaded ESMC model.
    max_len
        Maximum ESM input length, including BOS and EOS tokens.
    feature_dim
        Optional number of embedding dimensions to retain.

    Returns
    -------
    list
        Tuples containing the truncated sequence, padded embedding matrix, and
        attention mask.
    """
    results = []
    max_residues = max_len - 2  # Reserve positions for BOS and EOS tokens.

    for sequence in sequences:
        sequence = str(sequence)[:max_residues]
        protein = ESMProtein(sequence=sequence)

        with torch.no_grad():
            protein_tensor = model.encode(protein)
            logits_output = model.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True),
            )

            embeddings = logits_output.embeddings[0].cpu()

            # Remove BOS and EOS embeddings.
            real_length = len(sequence)
            protein_feature = embeddings[1 : real_length + 1]

            embedding_dim = protein_feature.shape[1]
            if feature_dim is not None and embedding_dim > feature_dim:
                protein_feature = protein_feature[:, :feature_dim]
                embedding_dim = feature_dim

            pad_len = max_residues - real_length
            if pad_len > 0:
                pad_tensor = torch.zeros(pad_len, embedding_dim)
                protein_feature = torch.cat([protein_feature, pad_tensor], dim=0)

            attention_mask = torch.cat(
                [torch.ones(real_length), torch.zeros(pad_len)]
            ).long()

        results.append((sequence, protein_feature.numpy(), attention_mask.numpy()))

    return results


def read_plddt_low_confidence_indices(
    plddt_path: Path | None,
    plddt_threshold: float,
) -> np.ndarray | None:
    """Read pLDDT confidence scores and return low-confidence residue indices."""
    if plddt_path is None:
        return None

    if not plddt_path.exists():
        LOGGER.warning("pLDDT file not found: %s", plddt_path)
        return None

    try:
        with plddt_path.open("r", encoding="utf-8") as handle:
            plddts = np.array(json.load(handle)["confidenceScore"])
        return np.where(plddts < plddt_threshold)[0]
    except Exception as exc:
        LOGGER.warning("Failed to read pLDDT file %s: %s", plddt_path, exc)
        return None


def get_structure_sequence(
    foldseek: str,
    cif_path: Path,
    chains: list[str] | None = None,
    plddt_path: Path | None = None,
    plddt_threshold: float = 70.0,
    threads: int = 1,
) -> dict[str, tuple[str, str, str]]:
    """
    Extract Foldseek 3Di structural sequences from a CIF structure.

    Returns
    -------
    dict
        Mapping from chain identifier to a tuple containing amino-acid sequence,
        3Di structural sequence, and combined SaProt input sequence.
    """
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    low_confidence_indices = read_plddt_low_confidence_indices(
        plddt_path=plddt_path,
        plddt_threshold=plddt_threshold,
    )

    sequence_dict: dict[str, tuple[str, str, str]] = {}

    with tempfile.TemporaryDirectory(prefix="foldseek_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        output_base = tmp_path / "get_struc_seq"

        command = [
            foldseek,
            "structureto3didescriptor",
            "-v",
            "0",
            "--threads",
            str(threads),
            "--chain-name-mode",
            "1",
            str(cif_path),
            str(output_base),
        ]

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if process.returncode != 0:
            LOGGER.warning(
                "Foldseek failed for %s: %s",
                cif_path,
                process.stderr.decode(errors="ignore"),
            )
            return {}

        for output_file in tmp_path.iterdir():
            if not output_file.name.startswith(output_base.name):
                continue
            if output_file.name.endswith(".dbtype"):
                continue

            with output_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue

                    description, sequence, structural_sequence = parts[:3]

                    if low_confidence_indices is not None and structural_sequence:
                        structural_array = np.array(list(structural_sequence))
                        valid_indices = low_confidence_indices[
                            low_confidence_indices < len(structural_array)
                        ]
                        structural_array[valid_indices] = "#"
                        structural_sequence = "".join(structural_array)

                    name_chain = description.split(" ")[0]
                    chain = name_chain.split("_")[-1] if "_" in name_chain else ""

                    if chains is not None and chain not in chains:
                        continue

                    if chain not in sequence_dict:
                        combined_sequence = "".join(
                            residue + structural_token.lower()
                            for residue, structural_token in zip(
                                sequence, structural_sequence
                            )
                        )
                        sequence_dict[chain] = (
                            sequence,
                            structural_sequence,
                            combined_sequence,
                        )

    return sequence_dict


def extract_saprot_sequences_from_ids(
    uniprot_ids: Iterable[str],
    cif_dir: Path,
    foldseek: str,
    plddt_threshold: float = 70.0,
    threads: int = 1,
) -> dict[str, tuple[str, str, str]]:
    """
    Extract SaProt-compatible sequence representations from CIF files.

    Chains A-Z are searched, and the first successfully parsed chain is used for
    each UniProt identifier.
    """
    saprot_sequence_dict: dict[str, tuple[str, str, str]] = {}

    for uniprot_id in sorted(set(map(str, uniprot_ids))):
        cif_path = cif_dir / f"{uniprot_id}.cif"

        if not cif_path.exists():
            LOGGER.warning("CIF file not found for %s: %s", uniprot_id, cif_path)
            continue

        for chain in [chr(i) for i in range(ord("A"), ord("Z") + 1)]:
            try:
                parsed = get_structure_sequence(
                    foldseek=foldseek,
                    cif_path=cif_path,
                    chains=[chain],
                    plddt_threshold=plddt_threshold,
                    threads=threads,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Could not parse %s chain %s: %s", cif_path.name, chain, exc
                )
                continue

            if chain in parsed:
                saprot_sequence_dict[uniprot_id] = parsed[chain]
                break

    return saprot_sequence_dict


def insert_saprot_sequence_columns(
    dataframe: pd.DataFrame,
    saprot_sequence_dict: dict[str, tuple[str, str, str]],
) -> pd.DataFrame:
    """Add SaProt sequence, structural sequence, and combined sequence columns."""
    output = dataframe.copy()
    output["SA_seq"] = output["uniprot_id"].map(
        lambda value: saprot_sequence_dict.get(str(value), (None, None, None))[0]
    )
    output["SA_struc_seq"] = output["uniprot_id"].map(
        lambda value: saprot_sequence_dict.get(str(value), (None, None, None))[1]
    )
    output["SA_combined_seq"] = output["uniprot_id"].map(
        lambda value: saprot_sequence_dict.get(str(value), (None, None, None))[2]
    )
    return output


def split_residue_structure_pairs(sequence: str) -> list[str]:
    """Split a SaProt combined sequence into residue-structure token pairs."""
    sequence = str(sequence)
    if len(sequence) % 2 != 0:
        sequence = sequence[:-1]
    return [sequence[i : i + 2] for i in range(0, len(sequence), 2)]


@torch.no_grad()
def word_to_single_id_or_unknown(word: str, tokenizer: AutoTokenizer) -> int:
    """Convert a residue-structure token pair to a tokenizer ID."""
    encoded = tokenizer(
        [word],
        is_split_into_words=True,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_tensors=None,
    )

    token_ids = encoded["input_ids"][0]
    if isinstance(token_ids, (int, np.integer)):
        token_ids = [int(token_ids)]
    else:
        token_ids = list(token_ids)

    if len(token_ids) == 1:
        return int(token_ids[0])

    return int(tokenizer.unk_token_id)


@torch.no_grad()
def pairs_batch_to_fixed_ids(
    pairs_batch: list[list[str]],
    tokenizer: AutoTokenizer,
    max_pairs: int = 1022,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Convert residue-structure token pairs to fixed-length token IDs.

    Parameters
    ----------
    pairs_batch
        Batch of tokenised protein sequences represented as residue-structure
        token pairs.
    tokenizer
        SaProt tokenizer.
    max_pairs
        Maximum number of residue-structure token pairs.

    Returns
    -------
    input_ids
        Token IDs padded to ``max_pairs``.
    attention_mask
        Binary mask indicating valid token positions.
    lengths
        Number of valid residue-structure token pairs for each sequence.
    """
    pad_id = int(tokenizer.pad_token_id)
    batch_size = len(pairs_batch)

    input_ids = np.full((batch_size, max_pairs), pad_id, dtype=np.int32)
    attention_mask = np.zeros((batch_size, max_pairs), dtype=np.uint8)
    lengths = []

    for row_index, pairs in enumerate(pairs_batch):
        token_ids = [
            word_to_single_id_or_unknown(token, tokenizer)
            for token in pairs[:max_pairs]
        ]

        valid_length = len(token_ids)
        input_ids[row_index, :valid_length] = token_ids
        attention_mask[row_index, :valid_length] = 1
        lengths.append(valid_length)

    return input_ids, attention_mask, lengths


@torch.no_grad()
def saprot_forward_from_ids(
    input_ids_np: np.ndarray,
    attention_mask_np: np.ndarray,
    model: AutoModel,
    device: torch.device | None = None,
) -> np.ndarray:
    """Run SaProt on fixed-length token IDs."""
    if device is None:
        device = next(model.parameters()).device

    input_ids = torch.as_tensor(input_ids_np, device=device, dtype=torch.long)
    attention_mask = torch.as_tensor(
        attention_mask_np, device=device, dtype=torch.long
    )

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    hidden = output.last_hidden_state
    hidden = hidden.masked_fill((attention_mask == 0).unsqueeze(-1), 0.0)

    return hidden.detach().cpu().to(torch.float16).numpy()


def add_saprot_embeddings_to_dataframe(
    dataframe: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int = 32,
    max_len: int = 1022,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Generate SaProt embeddings and add them to a dataframe."""
    if device is None:
        device = next(model.parameters()).device

    embeddings = []
    attention_masks = []
    input_ids_list = []
    valid_lengths = []

    sequences = dataframe["SA_combined_seq"].fillna("").astype(str).tolist()

    for batch_start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[batch_start : batch_start + batch_size]
        paired_sequences = [
            split_residue_structure_pairs(sequence[: max_len * 2])
            for sequence in batch_sequences
        ]

        input_ids_np, attention_mask_np, lengths = pairs_batch_to_fixed_ids(
            paired_sequences,
            tokenizer,
            max_pairs=max_len,
        )

        hidden = saprot_forward_from_ids(
            input_ids_np=input_ids_np,
            attention_mask_np=attention_mask_np,
            model=model,
            device=device,
        )

        for row_index in range(hidden.shape[0]):
            embeddings.append(hidden[row_index])
            attention_masks.append(attention_mask_np[row_index].astype(np.uint8))
            input_ids_list.append(input_ids_np[row_index].astype(np.int32))
            valid_lengths.append(int(lengths[row_index]))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = dataframe.copy()
    output["SA_embedding"] = embeddings
    output["SA_attention_mask"] = attention_masks
    output["SA_input_ids"] = input_ids_list
    output["SA_valid_len"] = valid_lengths

    return output


def infer_input_paths(dataset: str, seed: str, input_dir: Path) -> tuple[Path, Path, Path]:
    """Infer DrugBAN-style input paths from dataset, seed, and input directory."""
    if seed == "drugban":
        return (
            input_dir / "source_train.csv",
            input_dir / "target_train.csv",
            input_dir / "target_test.csv",
        )

    return (
        input_dir / f"source_train_{dataset}{seed}.csv",
        input_dir / f"target_train_{dataset}{seed}.csv",
        input_dir / f"target_test_{dataset}{seed}.csv",
    )


def add_sample_ids(dataframe: pd.DataFrame, prefix: str = "S") -> pd.DataFrame:
    """Add a Sample_id column if it is not already present."""
    output = dataframe.copy()
    if "Sample_id" not in output.columns:
        output.insert(0, "Sample_id", [f"{prefix}{i + 1}" for i in range(len(output))])
    return output


def truncate_sa_sequences(dataframe: pd.DataFrame, max_residues: int) -> pd.DataFrame:
    """Truncate SA_seq values to the maximum model length."""
    output = dataframe.copy()
    output["SA_seq"] = output["SA_seq"].fillna("").astype(str).str[:max_residues]
    return output


def merge_esmc_embeddings(
    dataframe: pd.DataFrame,
    model: ESMC,
    feature_dim: int,
    max_len: int,
) -> pd.DataFrame:
    """Generate ESMC embeddings for unique protein sequences and merge them."""
    protein_sequences = dataframe["SA_seq"].dropna().unique()
    protein_features = get_protein_features_token_level(
        protein_sequences,
        model=model,
        max_len=max_len,
        feature_dim=feature_dim,
    )

    protein_dataframe = pd.DataFrame(
        protein_features,
        columns=["SA_seq", "esm_tokens", "esm_mask"],
    )

    return pd.merge(dataframe, protein_dataframe, on="SA_seq", how="left")


def validate_required_columns(dataframe: pd.DataFrame, path: Path) -> None:
    """Check that the input dataframe contains required columns."""
    required_columns = {"uniprot_id"}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing_columns)}"
        )


def process_dataset(args: argparse.Namespace) -> None:
    """Run the full preprocessing workflow."""
    foldseek_bin = resolve_foldseek(args.foldseek)

    input_dir = Path(args.input_dir)
    cif_dir = Path(args.cif_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path, val_path, test_path = infer_input_paths(
        dataset=args.dataset,
        seed=args.seed,
        input_dir=input_dir,
    )

    LOGGER.info("Loading input partitions.")
    train_dataframe = pd.read_csv(train_path)
    val_dataframe = pd.read_csv(val_path)
    test_dataframe = pd.read_csv(test_path)

    for dataframe, path in [
        (train_dataframe, train_path),
        (val_dataframe, val_path),
        (test_dataframe, test_path),
    ]:
        validate_required_columns(dataframe, path)

    train_dataframe = add_sample_ids(train_dataframe)
    val_dataframe = add_sample_ids(val_dataframe)
    test_dataframe = add_sample_ids(test_dataframe)

    all_uniprot_ids = (
        set(train_dataframe["uniprot_id"])
        | set(val_dataframe["uniprot_id"])
        | set(test_dataframe["uniprot_id"])
    )

    LOGGER.info("Extracting Foldseek/3Di structural sequences.")
    saprot_sequence_dict = extract_saprot_sequences_from_ids(
        uniprot_ids=all_uniprot_ids,
        cif_dir=cif_dir,
        foldseek=foldseek_bin,
        plddt_threshold=args.plddt_threshold,
        threads=args.foldseek_threads,
    )

    train_dataframe = insert_saprot_sequence_columns(
        train_dataframe, saprot_sequence_dict
    )
    val_dataframe = insert_saprot_sequence_columns(val_dataframe, saprot_sequence_dict)
    test_dataframe = insert_saprot_sequence_columns(
        test_dataframe, saprot_sequence_dict
    )

    train_dataframe = truncate_sa_sequences(train_dataframe, args.max_residues)
    val_dataframe = truncate_sa_sequences(val_dataframe, args.max_residues)
    test_dataframe = truncate_sa_sequences(test_dataframe, args.max_residues)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    LOGGER.info("Loading ESMC model: %s", args.esmc_model)
    esmc_model = ESMC.from_pretrained(args.esmc_model).to(device)

    LOGGER.info("Generating ESMC embeddings.")
    train_dataframe = merge_esmc_embeddings(
        train_dataframe,
        model=esmc_model,
        feature_dim=args.esmc_feature_dim,
        max_len=args.esmc_max_len,
    )
    val_dataframe = merge_esmc_embeddings(
        val_dataframe,
        model=esmc_model,
        feature_dim=args.esmc_feature_dim,
        max_len=args.esmc_max_len,
    )
    test_dataframe = merge_esmc_embeddings(
        test_dataframe,
        model=esmc_model,
        feature_dim=args.esmc_feature_dim,
        max_len=args.esmc_max_len,
    )
    LOGGER.info("ESMC feature extraction completed.")

    LOGGER.info("Loading SaProt model: %s", args.saprot_model)
    tokenizer = AutoTokenizer.from_pretrained(args.saprot_model)
    saprot_model = AutoModel.from_pretrained(args.saprot_model).to(device).eval()

    LOGGER.info("Generating SaProt embeddings.")
    train_dataframe = add_saprot_embeddings_to_dataframe(
        train_dataframe,
        tokenizer,
        saprot_model,
        batch_size=args.batch_size,
        max_len=args.max_residues,
        device=device,
    )
    val_dataframe = add_saprot_embeddings_to_dataframe(
        val_dataframe,
        tokenizer,
        saprot_model,
        batch_size=args.batch_size,
        max_len=args.max_residues,
        device=device,
    )
    test_dataframe = add_saprot_embeddings_to_dataframe(
        test_dataframe,
        tokenizer,
        saprot_model,
        batch_size=args.batch_size,
        max_len=args.max_residues,
        device=device,
    )
    LOGGER.info("SaProt feature extraction completed.")

    output_paths = {
        "source_train": output_dir / args.train_output,
        "target_train": output_dir / args.val_output,
        "target_test": output_dir / args.test_output,
    }

    LOGGER.info("Saving processed datasets.")
    with output_paths["source_train"].open("wb") as handle:
        pickle.dump(train_dataframe, handle)

    with output_paths["target_train"].open("wb") as handle:
        pickle.dump(val_dataframe, handle)

    with output_paths["target_test"].open("wb") as handle:
        pickle.dump(test_dataframe, handle)

    for split_name, output_path in output_paths.items():
        LOGGER.info("Saved %s split: %s", split_name, output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate ESMC and SaProt features for DrugBAN-style CPI datasets."
        )
    )

    parser.add_argument(
        "--dataset",
        default="bindingdb",
        choices=["bindingdb", "biosnap"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--seed",
        default="drugban",
        help='Split identifier. Use "drugban" for the predefined DrugBAN split.',
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing source_train/target_train/target_test CSV files.",
    )
    parser.add_argument(
        "--cif-dir",
        required=True,
        help="Directory containing CIF files named as <uniprot_id>.cif.",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="Directory where processed pickle files will be saved.",
    )
    parser.add_argument(
        "--foldseek",
        default=None,
        help="Path to Foldseek executable. If omitted, PATH or FOLDSEEK is used.",
    )
    parser.add_argument(
        "--foldseek-threads",
        type=int,
        default=1,
        help="Number of Foldseek threads.",
    )
    parser.add_argument(
        "--plddt-threshold",
        type=float,
        default=70.0,
        help="Residues below this pLDDT threshold are marked as low confidence.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for SaProt embedding generation.",
    )
    parser.add_argument(
        "--max-residues",
        type=int,
        default=1022,
        help="Maximum number of residues retained for SaProt input.",
    )
    parser.add_argument(
        "--esmc-model",
        default="esmc_600m",
        help="ESMC model identifier.",
    )
    parser.add_argument(
        "--esmc-max-len",
        type=int,
        default=1024,
        help="Maximum ESMC model length, including BOS and EOS tokens.",
    )
    parser.add_argument(
        "--esmc-feature-dim",
        type=int,
        default=1152,
        help="Number of ESMC embedding dimensions retained.",
    )
    parser.add_argument(
        "--saprot-model",
        default="westlake-repl/SaProt_650M_AF2",
        help="Hugging Face identifier for the SaProt model.",
    )
    parser.add_argument(
        "--train-output",
        default="source_train_processed_saprot.pkl",
        help="Filename for the processed source_train split.",
    )
    parser.add_argument(
        "--val-output",
        default="target_train_processed_saprot.pkl",
        help="Filename for the processed target_train split.",
    )
    parser.add_argument(
        "--test-output",
        default="target_test_processed_saprot.pkl",
        help="Filename for the processed target_test split.",
    )

    return parser


def main() -> None:
    """Entry point."""
    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args()
    process_dataset(args)


if __name__ == "__main__":
    main()

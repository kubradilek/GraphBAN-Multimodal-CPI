#!/usr/bin/env python3
"""
Prediction and attention extraction for the multimodal GraphBAN-style CPI model.

This script loads a processed compound-protein interaction dataset and a trained
checkpoint, generates prediction probabilities, and exports bilinear attention maps
for correctly predicted positive interactions. It is intended for use after running
feature extraction and model training.

Expected processed input columns include:
    - Sample_id (optional; generated if absent)
    - SMILES
    - SA_seq
    - esm_tokens
    - esm_mask
    - SA_embedding
    - Y (optional; used for evaluation/attention filtering when available)

Author: Kubra Dilek Babaarslan
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dgllife.model import GCN as DGL_GCN
from dgllife.utils import (
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    smiles_to_bigraph,
)
from rdkit import Chem
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset
from tqdm import tqdm

from att_BANmask import BANLayer


LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Model components
# -----------------------------------------------------------------------------
class InteractionBAN(nn.Module):
    """Bilinear attention fusion module for protein and compound features."""

    def __init__(
        self,
        dim_esm: int = 2432,
        dim_drug: int = 128,
        hidden: int = 128,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.ban_layer = weight_norm(
            BANLayer(v_dim=dim_drug, q_dim=dim_esm, h_dim=hidden, h_out=2),
            name="h_mat",
            dim=None,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
        self.output_dim = out_dim

    def forward(
        self,
        protein_tokens: torch.Tensor,
        drug_tokens: torch.Tensor,
        protein_mask: torch.Tensor,
        drug_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fused, attention = self.ban_layer(
            drug_tokens,
            protein_tokens,
            v_mask=drug_mask,
            q_mask=protein_mask,
            softmax=True,
        )
        return self.proj(fused), attention


class Classifier(nn.Module):
    """Binary classifier applied to the fused CPI representation."""

    def __init__(self, input_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h).view(-1)


class MolecularGCN(nn.Module):
    """GCN-based molecular graph encoder."""

    def __init__(
        self,
        dim_embedding: int = 128,
        hidden_feats: Optional[List[int]] = None,
        activation: Optional[List[Any]] = None,
    ) -> None:
        super().__init__()
        if hidden_feats is None:
            hidden_feats = [128, 128, 128]
        if activation is None:
            activation = [nn.functional.relu] * len(hidden_feats)

        self.init_transform = nn.LazyLinear(dim_embedding, bias=False)
        self.gnn = DGL_GCN(
            in_feats=dim_embedding,
            hidden_feats=hidden_feats,
            activation=activation,
        )
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph: dgl.DGLGraph, mask: torch.Tensor) -> torch.Tensor:
        node_features = batch_graph.ndata["h"]
        node_features = self.init_transform(node_features)
        node_features = self.gnn(batch_graph, node_features)

        batch_size, max_nodes = mask.shape
        output = node_features.new_zeros(batch_size, max_nodes, self.output_feats)
        pointer = 0
        node_counts = batch_graph.batch_num_nodes().tolist()

        for batch_index, node_count in enumerate(node_counts):
            retained_nodes = min(node_count, max_nodes)
            output[batch_index, :retained_nodes, :] = node_features[
                pointer : pointer + retained_nodes, :
            ]
            pointer += node_count

        return output


class DrugFeature(nn.Module):
    """Wrapper around the molecular GCN encoder."""

    def __init__(self, dim_embedding: int = 128, hidden_feats: Optional[List[int]] = None):
        super().__init__()
        if hidden_feats is None:
            hidden_feats = [128, 128, 128]
        self.drug_extractor = MolecularGCN(
            dim_embedding=dim_embedding,
            hidden_feats=hidden_feats,
        )
        self.output_feats = self.drug_extractor.output_feats

    def forward(self, drug_graph: dgl.DGLGraph, drug_mask: torch.Tensor) -> torch.Tensor:
        return self.drug_extractor(drug_graph, drug_mask)


# -----------------------------------------------------------------------------
# Dataset and utilities
# -----------------------------------------------------------------------------
class TestGraphDataset(Dataset):
    """Dataset for prediction and attention extraction."""

    def __init__(self, df: pd.DataFrame, max_drug_nodes: int = 290) -> None:
        self.df = df.reset_index(drop=True)
        self.max_drug_nodes = max_drug_nodes
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.graph_builder = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        esm = torch.as_tensor(row["esm_tokens"], dtype=torch.float32)
        saprot = torch.as_tensor(row["SA_embedding"], dtype=torch.float32)
        protein_features = torch.cat([esm, saprot], dim=1)
        protein_mask = torch.as_tensor(row["esm_mask"], dtype=torch.float32)

        smiles = row["SMILES"]
        graph = self.graph_builder(
            smiles=smiles,
            node_featurizer=self.atom_featurizer,
            edge_featurizer=self.bond_featurizer,
        )

        num_nodes = graph.num_nodes()
        retained_nodes = min(num_nodes, self.max_drug_nodes)
        drug_mask = torch.zeros(self.max_drug_nodes, dtype=torch.float32)
        drug_mask[:retained_nodes] = 1.0

        mol = Chem.MolFromSmiles(smiles) if smiles is not None else None
        if mol is not None:
            atom_tokens = [
                f"{atom.GetSymbol()}{atom.GetAtomicNum()}_#{atom.GetIdx()}"
                for atom in mol.GetAtoms()
            ]
        else:
            atom_tokens = [f"X_#{i}" for i in range(num_nodes)]

        graph.ndata["atom_idx"] = torch.arange(num_nodes, dtype=torch.long)

        sample_id = row.get("Sample_id", f"S{idx + 1}")
        protein_sequence = row["SA_seq"]
        y_true = row.get("Y", None)

        return (
            protein_features,
            protein_mask,
            graph,
            drug_mask,
            sample_id,
            protein_sequence,
            atom_tokens,
            y_true,
        )


def load_checkpoint(path: Path, map_location: torch.device) -> Dict[str, Any]:
    """Load a PyTorch checkpoint while supporting multiple PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_attention_csv(
    attention: torch.Tensor,
    protein_tokens: List[str],
    atom_tokens: List[str],
    protein_mask: Optional[torch.Tensor],
    drug_mask: Optional[torch.Tensor],
    output_prefix: Path,
) -> None:
    """Save an atom-by-residue attention matrix as a CSV file."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    attention_matrix = attention.squeeze(0).sum(0).detach().cpu().numpy()

    if drug_mask is not None:
        keep_atoms = drug_mask.bool().cpu().numpy()
        attention_matrix = attention_matrix[keep_atoms, :]
        atom_tokens = [token for token, keep in zip(atom_tokens, keep_atoms) if keep]

    if protein_mask is not None:
        keep_residues = protein_mask.bool().cpu().numpy()
        attention_matrix = attention_matrix[:, keep_residues]
        protein_tokens = [
            token for token, keep in zip(protein_tokens, keep_residues) if keep
        ]

    attention_df = pd.DataFrame(
        attention_matrix,
        index=atom_tokens,
        columns=protein_tokens,
    )
    output_file = output_prefix.with_name(output_prefix.name + "ban_attention.csv")
    attention_df.to_csv(output_file)
    LOGGER.info("Saved attention matrix: %s", output_file)


def get_config_from_checkpoint(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model configuration from a saved checkpoint bundle."""
    best_trial = bundle.get("best_trial", None)
    config = getattr(best_trial, "user_attrs", {}).get("best_config", None)

    if not isinstance(config, dict):
        config = bundle.get("config", None)

    if not isinstance(config, dict):
        raise RuntimeError("Model configuration was not found in the checkpoint.")

    return config


# -----------------------------------------------------------------------------
# Prediction workflow
# -----------------------------------------------------------------------------
def run_prediction(
    processed_test_pkl: Path,
    checkpoint_path: Path,
    output_dir: Path,
    probability_threshold: float = 0.5,
    max_drug_nodes: int = 290,
) -> None:
    """Run prediction and export attention maps for correctly predicted positives."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    with processed_test_pkl.open("rb") as handle:
        df_test = pickle.load(handle)

    if "Sample_id" not in df_test.columns:
        df_test.insert(0, "Sample_id", [f"S{i + 1}" for i in range(len(df_test))])

    checkpoint = load_checkpoint(checkpoint_path, device)
    config = get_config_from_checkpoint(checkpoint)

    hidden_dim = int(config["hidden_dim"])
    out_dim = int(config["out_dim"])
    dropout = float(config.get("dropout", 0.2))

    interaction_model = InteractionBAN(hidden=hidden_dim, out_dim=out_dim).to(device).eval()
    classifier = Classifier(input_dim=out_dim, dropout=dropout).to(device).eval()
    drug_feature = DrugFeature(dim_embedding=128, hidden_feats=[128, 128, 128]).to(device).eval()

    required_keys = {"interaction_model", "classifier", "drug_feature"}
    if not required_keys.issubset(checkpoint.keys()):
        missing = required_keys.difference(checkpoint.keys())
        raise RuntimeError(f"Checkpoint is missing required keys: {sorted(missing)}")

    interaction_model.load_state_dict(checkpoint["interaction_model"])
    classifier.load_state_dict(checkpoint["classifier"])
    drug_feature.load_state_dict(checkpoint["drug_feature"])

    dataset = TestGraphDataset(df_test, max_drug_nodes=max_drug_nodes)

    y_probs: List[float] = []
    y_preds: List[int] = []
    y_trues: List[Optional[int]] = []
    sample_ids: List[str] = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Predicting and extracting attention"):
            (
                protein_features,
                protein_mask,
                graph,
                drug_mask,
                sample_id,
                protein_sequence,
                atom_tokens,
                y_true,
            ) = dataset[i]

            protein_features = protein_features.unsqueeze(0).to(device)
            protein_mask = protein_mask.unsqueeze(0).to(device)
            graph = graph.to(device)
            drug_mask = drug_mask.unsqueeze(0).to(device)

            drug_tokens = drug_feature(graph, drug_mask)
            fused, attention = interaction_model(
                protein_features,
                drug_tokens,
                protein_mask,
                drug_mask,
            )
            logit = classifier(fused).view(-1)
            probability = torch.sigmoid(logit).item()
            predicted_label = int(probability >= probability_threshold)

            y_probs.append(probability)
            y_preds.append(predicted_label)
            sample_ids.append(sample_id)

            if y_true is None or (isinstance(y_true, float) and np.isnan(y_true)):
                y_trues.append(None)
            else:
                y_trues.append(int(y_true))

            if y_true is not None and int(y_true) == 1 and predicted_label == 1:
                protein_tokens = list(protein_sequence) if isinstance(protein_sequence, str) else []
                output_prefix = output_dir / f"{sample_id}_"
                save_attention_csv(
                    attention=attention,
                    protein_tokens=protein_tokens,
                    atom_tokens=atom_tokens,
                    protein_mask=protein_mask[0],
                    drug_mask=drug_mask[0],
                    output_prefix=output_prefix,
                )

    result_df = pd.DataFrame(
        {
            "Sample_id": sample_ids,
            "Pred_Prob": y_probs,
            "Pred_Label": y_preds,
            "Y_true": y_trues,
        }
    )
    merged = df_test.merge(result_df, on="Sample_id", how="left")

    if merged["Y_true"].notna().sum() > 1:
        merged["Pred_Label_F1"] = (
            merged["Pred_Prob"] >= probability_threshold
        ).astype(int)
        LOGGER.info("Prediction threshold used: %.4f", probability_threshold)

    output_file = output_dir / "test_with_predictions.csv"
    merged.to_csv(output_file, index=False)
    LOGGER.info("Saved prediction results: %s", output_file)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run CPI prediction and extract BAN attention maps."
    )
    parser.add_argument(
        "--processed-test-pkl",
        required=True,
        type=Path,
        help="Path to the processed target_test pickle file.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to the trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/attention"),
        type=Path,
        help="Directory for prediction results and attention matrices.",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Probability threshold for binary classification.",
    )
    parser.add_argument(
        "--max-drug-nodes",
        default=290,
        type=int,
        help="Maximum number of compound graph nodes retained per sample.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    run_prediction(
        processed_test_pkl=args.processed_test_pkl,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        probability_threshold=args.threshold,
        max_drug_nodes=args.max_drug_nodes,
    )


if __name__ == "__main__":
    main()

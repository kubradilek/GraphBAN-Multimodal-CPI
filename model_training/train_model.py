import os
import random
import sys
import copy
import pathlib
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.weight_norm import weight_norm
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             confusion_matrix, precision_recall_curve, precision_score,
                             f1_score, recall_score, accuracy_score)
import optuna
import pickle
from att_BANmask import BANLayer
from functools import partial
import dgl
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


set_seed(0)

class DTIDataset(Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        row_idx = self.list_IDs[index]
        row = self.df.iloc[row_idx]  

        # protein
        esm = torch.as_tensor(row['esm_tokens'], dtype=torch.float32)  # [Lp, 1152]
        sa  = torch.as_tensor(row['SA_embedding'],  dtype=torch.float32)  # [Lp, 1280]
        pro = torch.cat([esm, sa], dim=1)                               # [Lp, 2432]
        pro_mask = torch.as_tensor(row['esm_mask'], dtype=torch.float32)  # [Lp]

        # drug graph
        smi = row['SMILES']
        g = self.fc(
            smiles=smi,
            node_featurizer=self.atom_featurizer,
            edge_featurizer=self.bond_featurizer
        )

        n = g.num_nodes()
        L = self.max_drug_nodes
        take = min(n, L)
        v_d_mask = torch.zeros(L, dtype=torch.float32)
        v_d_mask[:take] = 1.0

        # label
        label = torch.tensor(row['Y'], dtype=torch.float32)

        return pro, g, pro_mask, v_d_mask, label


def collate_fn(batch, max_drug_nodes=290):
    pros, graphs, pro_masks, drug_masks, labels = zip(*batch)
    Lp_max = max(p.shape[0] for p in pros)
    pros_pad = []
    pro_masks_pad = []
    for p, m in zip(pros, pro_masks):
        pros_pad.append(F.pad(p, (0, 0, 0, Lp_max - p.shape[0])))
        pro_masks_pad.append(F.pad(m, (0, Lp_max - m.shape[0]), value=0))
    pros_pad = torch.stack(pros_pad)           # [B, Lp_max, 2432]
    pro_masks_pad = torch.stack(pro_masks_pad) # [B, Lp_max]

    bg = dgl.batch(graphs)                   

    drug_masks = torch.stack(drug_masks)       # [B, max_drug_nodes]
    labels = torch.stack(labels)               # [B]

    return pros_pad, bg, pro_masks_pad, drug_masks, labels

from dgllife.model import GCN

class MolecularGCN(nn.Module):
    def __init__(self, dim_embedding=128, hidden_feats=None, activation=None):
        super().__init__()
        if hidden_feats is None:
            hidden_feats = [128, 128, 128]
        if activation is None:
            activation = [F.relu, F.relu, F.relu]
        self.init_transform = nn.LazyLinear(dim_embedding, bias=False)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph, mask):  # mask: [B, Ld_max]
        node_feats = batch_graph.ndata['h']          # [sumN, in_feats(atom fea)]
        node_feats = self.init_transform(node_feats) # -> [sumN, dim_embedding]
        node_feats = self.gnn(batch_graph, node_feats)  # [sumN, D]
        B, L = mask.shape
        out = node_feats.new_zeros(B, L, self.output_feats)
        ptr = 0
        ns = batch_graph.batch_num_nodes().tolist()
        for b, n in enumerate(ns):
            out[b, :n, :] = node_feats[ptr:ptr+n, :]
            ptr += n
        return out

class DrugFeature(nn.Module):
    def __init__(self, dim_embedding=128, hidden_feats=[128,128,128]):
        super().__init__()
        self.drug_extractor = MolecularGCN(dim_embedding=dim_embedding, hidden_feats=hidden_feats)
        self.output_feats = self.drug_extractor.output_feats

    def forward(self, bg_d, drug_mask):
        return self.drug_extractor(bg_d, drug_mask)


class InteractionBAN(nn.Module):
    def __init__(self, dim_esm=2432, dim_drug=128, hidden=128, out_dim=128):
        super().__init__()
        self.ban_layer = weight_norm(
            BANLayer(v_dim=dim_drug, q_dim=dim_esm, h_dim=hidden, h_out=2),
            name='h_mat', dim=None
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.output_dim = out_dim

    def forward(self, esm_tokens, drug_tokens, esm_mask, drug_mask):
        fused, att = self.ban_layer(
            drug_tokens, esm_tokens,
            v_mask=drug_mask, q_mask=esm_mask,
            softmax=True
        )
        return self.proj(fused), att


class Classifier(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),    
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, h):
        return self.head(h).view(-1)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=128, proj_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return self.proj(x)



# Contrastive Loss

def strict_supervised_contrastive_loss(h, labels, temperature=0.07):
    """
    h: [N, D]; labels: [N] in {0,1}
    Compute supervised contrastive loss using only positive pairs; the denominator is normalised over valid pairs.
    """
    h = F.normalize(h, dim=1)
    sim_matrix = torch.matmul(h, h.T) / temperature
    N = h.size(0)

    labels = labels.view(-1)
    label_i = labels.view(N, 1)
    label_j = labels.view(1, N)

    logits_mask = torch.ones((N, N), device=h.device) - torch.eye(N, device=h.device)
    pos_mask = ((label_i == 1) & (label_j == 1)).float() * logits_mask
    neg_mask = (label_i != label_j).float() * logits_mask
    valid_pair_mask = (pos_mask + neg_mask).clamp(max=1.0)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()
    exp_logits = torch.exp(logits) * valid_pair_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

    anchor_mask = ((pos_mask.sum(1) > 0) | (neg_mask.sum(1) > 0)).float()
    pos_sum = (pos_mask * log_prob).sum(1)
    pos_count = pos_mask.sum(1).clamp(min=1.0)
    per_sample_loss = - (pos_sum / pos_count)
    loss = (anchor_mask * per_sample_loss).sum() / anchor_mask.sum().clamp(min=1.0)
    return loss



# Diffusion

class DiffusionEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 3, max_t: int = 1000):
        super().__init__()
        self.max_t = max_t
        self.time_emb = nn.Embedding(max_t, embed_dim)
        layers = [nn.Linear(embed_dim * 2, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, embed_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t):
        t_emb = self.time_emb(t)
        h = torch.cat([x_t, t_emb], dim=1)
        return self.net(h)


def get_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    return torch.linspace(beta_start, beta_end, T)


class LatentDiffusionHelper:
    def __init__(self, T: int = 1000, device: torch.device = torch.device('cpu')):
        self.T = T
        betas = get_beta_schedule(T).to(device)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

    def q_sample(self, x0, t, noise):
        return self.sqrt_alpha_bar[t].unsqueeze(1) * x0 + self.sqrt_one_minus_alpha_bar[t].unsqueeze(1) * noise


def diffusion_x0_loss(diff_enc, helper, x0):
    N = x0.size(0)
    t = torch.randint(0, helper.T, (N,), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = helper.q_sample(x0, t, noise)
    x0_hat = diff_enc(x_t, t)
    return F.mse_loss(x0_hat, x0)



# Training Loop (joint)

def train_joint(interaction_model, proj_head, diff_enc, clf,
                drug_feature,                
                dataloader, opt_main, opt_diff, helper, criterion,
                alpha, beta, device, model_type='ban'):

    interaction_model.train(); proj_head.train(); diff_enc.train(); clf.train(); drug_feature.train()

    total_loss, total_diff = 0.0, 0.0
    for pro, bg,pro_mask, drug_mask, label in dataloader:
        pro, pro_mask = pro.to(device), pro_mask.to(device)           # [B, Lp, 2432], [B, Lp]
        bg = bg.to(device)                                            # batched DGLGraph
        drug_mask = drug_mask.to(device)                              # [B, Ld]
        label = label.to(device)

        #  GCN 
        v_d = drug_feature(bg, drug_mask)

        # BAN
        if model_type == 'mlp':
            h_cp = interaction_model(pro, v_d, pro_mask, drug_mask)
        else:
            h_cp, _ = interaction_model(pro, v_d, pro_mask, drug_mask)  # [B, out_dim]

        h_proj = proj_head(h_cp)
        logits = clf(h_cp)


        noise = torch.randn_like(h_cp)
        t = torch.randint(0, helper.T, (h_cp.size(0),), device=h_cp.device)
        x_t = helper.q_sample(h_cp, t, noise)
        h_cp_aug = diff_enc(x_t, t)
        h_proj_aug = proj_head(h_cp_aug)

        h_all = torch.cat([h_proj, h_proj_aug], dim=0)
        labels_all = torch.cat([label, label], dim=0)

        loss_bce = criterion(logits, label)
        loss_contrast = strict_supervised_contrastive_loss(h_all, labels_all) if alpha > 0 else torch.tensor(0.0, device=device)
        loss_diff = diffusion_x0_loss(diff_enc, helper, h_cp.detach()) if beta > 0 else torch.tensor(0.0, device=device)

        loss_main = loss_bce + alpha * loss_contrast
        loss_total = loss_main + beta * loss_diff

        opt_main.zero_grad(); opt_diff.zero_grad()
        loss_total.backward()
        opt_main.step(); opt_diff.step()

        total_loss += (loss_main.item() + beta * loss_diff.item()) * len(label)
        total_diff += loss_diff.item() * len(label)

    avg_diff = total_diff / len(dataloader.dataset)
    return total_loss / len(dataloader.dataset), avg_diff




# Metrics

def get_optimal_f1_threshold(y_true, y_probs, skip_head=5):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_curve = 2 * precision * recall / (precision + recall + 1e-8)
    if len(thresholds) <= skip_head:
        skip_head = 0
    f1_valid = f1_curve[skip_head:]
    if len(f1_valid) == 0:
        best_idx = np.argmax(f1_curve)
        return thresholds[best_idx if best_idx < len(thresholds) else -1]
    else:
        best_idx = np.argmax(f1_valid) + skip_head
        return thresholds[best_idx]


def evaluate(interaction_model, classifier, drug_feature, dataloader, device):
    interaction_model.eval(); classifier.eval(); drug_feature.eval()
    y_true, y_scores = [], []
    total_loss, total_samples = 0, 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for pro,  bg,pro_mask, drug_mask, label in dataloader:
            pro, pro_mask = pro.to(device), pro_mask.to(device)
            bg = bg.to(device)
            drug_mask = drug_mask.to(device)
            label = label.to(device)

            v_d = drug_feature(bg, drug_mask)  # [B, Ld, 128]
            h_cp, _ = interaction_model(pro, v_d, pro_mask, drug_mask)
            logits = classifier(h_cp).view(-1)
            loss = criterion(logits, label)

            total_loss += loss.item() * len(label)
            total_samples += len(label)
            y_true.extend(label.cpu().numpy().tolist())
            y_scores.extend(logits.cpu().numpy().tolist())

    test_loss = total_loss / max(total_samples, 1)
    y_probs = torch.sigmoid(torch.tensor(y_scores)).numpy()
    th = get_optimal_f1_threshold(y_true, y_probs)
    y_pred = (y_probs >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "auroc": roc_auc_score(y_true, y_probs),
        "auprc": average_precision_score(y_true, y_probs),
        "F1": f1_score(y_true, y_pred),
        "test_loss": test_loss,
        "sensitivity": recall_score(y_true, y_pred),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred)
    }



# Model Builder

def build_models(trial, device, model_type='ban'):
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    out_dim    = trial.suggest_categorical("out_dim",    [32, 64, 128, 256])
    proj_dim   = trial.suggest_categorical("proj_dim",   [32, 64, 128,256])
    dropout    = trial.suggest_categorical("dropout", [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])

    dim_drug = 128  # GCN 
    interaction_model = InteractionBAN(dim_esm=2432, dim_drug=dim_drug,
                                       hidden=hidden_dim, out_dim=out_dim).to(device)
    interaction_output_dim = interaction_model.output_dim

    projection_head = ProjectionHead(input_dim=interaction_output_dim, proj_dim=proj_dim).to(device)
    classifier = Classifier(input_dim=interaction_output_dim, dropout=dropout).to(device)

    
    drug_feature = DrugFeature(dim_embedding=128, hidden_feats=[128,128,128]).to(device) # DrugFeature

    diff_hidden_dim = trial.suggest_categorical("diff_hidden_dim", [32, 64, 128, 256])
    num_layers      = trial.suggest_int("num_layers", 2, 4)
    max_t           = trial.suggest_categorical("max_t", [750,800,850,900,950,1000,1050,1100,1150,1200,1250])

    diff_enc = DiffusionEncoder(
        embed_dim=interaction_output_dim,
        hidden_dim=diff_hidden_dim,
        num_layers=num_layers,
        max_t=max_t
    ).to(device)

    return interaction_model, projection_head, classifier, diff_enc, drug_feature, \
    interaction_output_dim, hidden_dim, out_dim, proj_dim, dropout



# Optuna Objective

def objective(trial, model_type='ban'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Hyperparameters (tighter) ===
    lr_main = trial.suggest_float("lr_main", 1e-4, 5e-4, log=True)
    lr_diff = trial.suggest_float("lr_diff", 1e-4, 5e-4, log=True)
    alpha   = trial.suggest_categorical("alpha",    [0.0, 0.05, 0.1, 0.15, 0.2,0.25, 0.3, 0.35,0.4, 0.5])
    beta_max= trial.suggest_categorical("beta_max", [0.0, 0.05, 0.1, 0.15, 0.2,0.25, 0.3])

    batch_size= 64
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True,
                          collate_fn=collate_fn)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False,
                          collate_fn=collate_fn)


    (interaction_model, projection_head, classifier, diff_enc, drug_feature,
 interaction_output_dim, hidden_dim, out_dim, proj_dim, dropout) = build_models(trial, device, model_type=model_type)

    opt_main = torch.optim.AdamW(
        list(interaction_model.parameters()) +
        list(projection_head.parameters()) +
        list(classifier.parameters()) +
        list(drug_feature.parameters()),           
        lr=lr_main
    )
    opt_diff = torch.optim.AdamW(diff_enc.parameters(), lr=lr_diff)
    scheduler_main = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_main, mode='max', patience=5)
    scheduler_diff = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_diff, mode='min', patience=5)

    criterion = nn.BCEWithLogitsLoss()
    helper = LatentDiffusionHelper(T=diff_enc.max_t, device=device)

    best_val_auroc, early_stop_counter, best_model_state, best_epoch = 0.0, 0, None, 0

    #Train
    for epoch in range(1, 150):
        beta = beta_max * min(1.0, epoch / 10)

        train_loss, avg_diff = train_joint(
            interaction_model, projection_head, diff_enc, classifier, drug_feature,
            train_loader, opt_main, opt_diff, helper,
            criterion, alpha, beta, device, model_type=model_type
        )
        val_metrics = evaluate(interaction_model, classifier, drug_feature, val_loader, device)
        scheduler_main.step(val_metrics['auroc'])
        scheduler_diff.step(avg_diff)

        # Optuna pruning hook
        trial.report(val_metrics['auroc'], epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            best_epoch = epoch
            best_model_state = {
                'drug_feature': copy.deepcopy(drug_feature.state_dict()),
                'interaction_model': copy.deepcopy(interaction_model.state_dict()),
                'classifier': copy.deepcopy(classifier.state_dict()),
                'config': {
                    'model_type': model_type,
                    'hidden_dim': hidden_dim,
                    'out_dim': out_dim,
                    'proj_dim': proj_dim,
                    'dropout': dropout,
                    'batch_size': batch_size,
                    'interaction_output_dim': interaction_output_dim
                }
            }
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= 20:
            break
    
    if best_model_state is None:
        best_model_state = {
            'drug_feature': copy.deepcopy(drug_feature.state_dict()),        
            'interaction_model': copy.deepcopy(interaction_model.state_dict()),
            'classifier': copy.deepcopy(classifier.state_dict()),
            'config': {
                'model_type': model_type,
                'hidden_dim': hidden_dim,
                'out_dim': out_dim,
                'proj_dim': proj_dim,
                'dropout': dropout,  
                'batch_size': batch_size,
                'interaction_output_dim': interaction_output_dim
            }
        }

    trial.set_user_attr("best_model_weights", {
        "drug_feature": best_model_state["drug_feature"],
        "interaction_model": best_model_state["interaction_model"],
        "classifier": best_model_state["classifier"],
    })
    trial.set_user_attr("best_config", best_model_state["config"])
    trial.set_user_attr("best_epoch", best_epoch)

    return best_val_auroc   



# Main  
if __name__ == '__main__':
    dataset = 'bindingdb'
    seed = str(sys.argv[1])

    if seed == "drugban":
        directory = f"/gpfs01/home/alykb3/GraphBAN_copy/Analysis/{dataset}/{seed}"
    else:
        directory = f"/gpfs01/home/alykb3/GraphBAN_copy/Analysis/{dataset}/seed{seed}"
  
    test_time =1
    with open(f'{directory}/df_train_final_DP2_SaProt.pkl', 'rb') as f:
        df_train = pickle.load(f)   

    with open(f'{directory}/df_val_final_DP2_SaProt.pkl', 'rb') as f:
        df_val = pickle.load(f)   

    with open(f'{directory}/df_test_final_DP2_SaProt.pkl', 'rb') as f:
        df_test = pickle.load(f)   
  
    ds_train = DTIDataset(df_train.index.values, df_train)
    ds_val = DTIDataset(df_val.index.values, df_val)
    ds_test = DTIDataset(df_test.index.values, df_test)

    n_trials_num = 100   
    model_type = 'ban'   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Optuna study (seeded TPE sampler with MedianPruner) ===
    set_seed(0)
    study = optuna.create_study(
        direction='maximize',   
        sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )

    # Enqueue a baseline trial using BCE loss only (alpha=0, beta_max=0).
    study.enqueue_trial({
        "lr_main": 3e-4, "lr_diff": 3e-4,
        "alpha": 0.0, "beta_max": 0.0,
        "hidden_dim": 256, "out_dim": 256, "proj_dim": 128,
        "dropout": 0.2,  
        "diff_hidden_dim": 128, "num_layers": 3, "max_t": 1000
    })

    study.optimize(lambda trial: objective(trial, model_type=model_type), n_trials=n_trials_num)

    print("Best hyperparameters:", study.best_params)
    best_trial = study.best_trial
    cfg = best_trial.user_attrs["best_config"]
    model_weights = best_trial.user_attrs["best_model_weights"]
    #Test loader
    batch_size = cfg['batch_size']
    loader_test = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, drop_last=False,
                             collate_fn=collate_fn)
    
    hidden_dim, out_dim = cfg["hidden_dim"], cfg["out_dim"]
    interaction_model = InteractionBAN(hidden=hidden_dim, out_dim=out_dim).to(device)
    classifier = Classifier(input_dim=out_dim, dropout=cfg["dropout"]).to(device)
    drug_feature = DrugFeature(dim_embedding=128, hidden_feats=[128,128,128]).to(device)
    
    interaction_model.load_state_dict(model_weights["interaction_model"])

    classifier.load_state_dict(model_weights["classifier"])
    drug_feature.load_state_dict(model_weights["drug_feature"])                        
    
    # Evaluate on Test
    test_metrics = evaluate(interaction_model, classifier, drug_feature, loader_test, device)
    test_metrics['best_epoch'] = best_trial.user_attrs.get("best_epoch", None)
    print("Final test set performance:", test_metrics)
    #  Save 
    pathlib.Path("checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({
        "interaction_model": interaction_model.state_dict(),
        "classifier": classifier.state_dict(),
        "drug_feature": drug_feature.state_dict(),      # Save drug feature extractor weights
        "best_trial": best_trial,
    }, f"checkpoints/{dataset}_seed{seed}_best_model_state_DP2_SaProt_test2_{test_time}.pt")

    try:
        joblib.dump(best_trial, f"checkpoints/{dataset}_seed{seed}_best_trial_DP2_SaProt_test2_{test_time}.pkl")
    except Exception:
        joblib.dump(study.best_params, f"checkpoints/{dataset}_seed{seed}_best_params_DP2_SaProt_test2_{test_time}.pkl")
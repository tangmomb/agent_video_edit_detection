"""Entraînement des classifieurs de train_model en PyTorch, sur GPU (ou CPU pour les tests) : mêmes
calculs que la chaîne scikit-learn de Lab._make, sans scikit-learn :
- StandardScaler (moyenne, écart-type ddof 0, écart nul -> 1) ;
- PCA (composantes principales exactes, centrées) ;
- LogisticRegression(C, class_weight="balanced") : même fonction objectif (perte logistique moyenne
  pondérée + ||w||² / (2 C somme des poids), intercept libre), L-BFGS en float64 ;
- MLPClassifier (relu, sortie logistique, alpha, Adam par défaut, lots de 200, mélange et
  initialisation tirés comme scikit-learn avec random_state=0), époque par époque avec l'arrêt de
  Lab._fit_mlp (AP par chaîne sur des chaînes mises de côté).

Les paramètres renvoyés servent au VPS à reconstruire des objets scikit-learn (remote_train.py) :
model.pkl et la production restent inchangés. Aucun import du projet : ce fichier est copié tel quel
dans l'image Modal (train_modal.py).
"""

from pathlib import Path

import numpy as np
import torch

LBFGS_MAX_ITER = 15000
# Gradient max de la fonction objectif moyenne (scikit-learn : 1e-4). Mesuré sur split_014 (4 000
# images, 8 embeddings) : AP stable dès 1e-5 ; 1e-6 = 160 itérations, 1e-7 = 342.
LBFGS_TOL = 1e-6
ADAM = {"lr": 1e-3, "beta_1": 0.9, "beta_2": 0.999, "eps": 1e-8}  # défauts de MLPClassifier
MLP_BATCH = 200


# ------------------------------------------------------------------ données
def load_matrix(root: Path, specs: list, sids: list[str], frames: list[int],
                threads: int = 1) -> tuple[np.ndarray | None, list]:
    """Matrice float32 (len(sids), dim) des embeddings en cache, alignée sur (sids, frames).
    specs : (dossier, parts, skip) par embedding : le fichier <root>/<dossier>/<sid>.npz contient ids
    (numéros d'images) et emb ; une vue dérivée (quads, tb) garde les parts après les skip premières
    (comme get_features). Renvoie (matrice, []) ou (None, [(dossier, sid), ...]) si des images
    manquent dans le cache. threads : fichiers lus en même temps (un volume Modal paie une latence
    réseau par fichier : 17 000 lectures une à une ne finissaient pas en 35 min)."""
    from concurrent.futures import ThreadPoolExecutor

    by_sample: dict[str, list[int]] = {}
    for k, s in enumerate(sids):
        by_sample.setdefault(s, []).append(k)
    frames = np.asarray(frames, dtype=np.int64)

    def read(dirname: str, s: str):
        path = Path(root) / dirname / f"{s}.npz"
        if not path.exists():
            return None
        with np.load(path) as z:
            ids, emb = z["ids"], z["emb"]
        want = frames[by_sample[s]]
        pos = np.searchsorted(ids, want)
        if (pos >= len(ids)).any() or (ids[np.minimum(pos, len(ids) - 1)] != want).any():
            return None
        return emb[pos].astype(np.float32, copy=False)

    mats, missing = [], []
    with ThreadPoolExecutor(max(1, threads)) as pool:
        for dirname, parts, skip in specs:
            out = None
            samples = list(by_sample)
            for s, sel in zip(samples, pool.map(lambda s: read(dirname, s), samples)):
                if sel is None:
                    missing.append((dirname, s))
                    continue
                if skip:
                    sel = sel[:, sel.shape[1] // parts * skip:]
                if out is None:
                    out = np.zeros((len(sids), sel.shape[1]), dtype=np.float32)
                out[by_sample[s]] = sel
            if out is not None:
                mats.append(out)
    if missing:
        return None, missing
    return np.concatenate(mats, axis=1), []


# ------------------------------------------------------------------ mesures
def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """average_precision_score de scikit-learn (cas binaire, ex aequo groupés)."""
    order = np.argsort(-p, kind="mergesort")
    y, p = np.asarray(y)[order].astype(float), np.asarray(p)[order]
    last = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    tps = np.cumsum(y)[last]
    if tps[-1] == 0:
        return 0.0
    precision = tps / (last + 1)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def group_ap(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> float | None:
    """metrics.group_ap : moyenne des AP des groupes qui ont les deux classes."""
    aps = [average_precision(y[m], p[m]) for m in (groups == g for g in np.unique(groups))
           if 0 < y[m].sum() < m.sum()]
    return float(np.mean(aps)) if aps else None


# ------------------------------------------------------------------ prétraitement
def _scaler(X: torch.Tensor) -> dict:
    mean = X.double().mean(0)
    var = ((X.double() - mean) ** 2).mean(0)
    scale = var.sqrt()
    scale[scale < 10 * torch.finfo(torch.float64).eps] = 1.0  # _handle_zeros_in_scale
    return {"mean": mean, "var": var, "scale": scale, "n": X.shape[0]}


def _pca(Z: torch.Tensor, k: int) -> dict:
    """Composantes principales exactes de Z (float64), par la plus petite des deux matrices de Gram."""
    n, d = Z.shape
    k = min(k, n, d)
    mean = Z.mean(0)
    Zc = (Z - mean).float()
    if d <= n:
        evals, evecs = torch.linalg.eigh(Zc.T @ Zc)
        top = torch.argsort(evals, descending=True)[:k]
        comps = evecs[:, top].T.double()
        sv = evals[top].clamp_min(0).sqrt().double()
    else:
        evals, evecs = torch.linalg.eigh(Zc @ Zc.T)
        top = torch.argsort(evals, descending=True)[:k]
        sv = evals[top].clamp_min(0).sqrt()
        comps = ((Zc.T @ evecs[:, top]) / sv.clamp_min(1e-12)).T.double()
        sv = sv.double()
        comps = comps / comps.norm(dim=1, keepdim=True).clamp_min(1e-12)
    # signes comme svd_flip de scikit-learn (u_based_decision=False) : la plus grande valeur
    # absolue de chaque composante est positive
    big = comps.abs().argmax(1)
    comps = comps * torch.sign(comps[torch.arange(k), big]).unsqueeze(1)
    total_var = ((Z - mean) ** 2).sum(0) / max(n - 1, 1)
    ev = sv ** 2 / max(n - 1, 1)
    return {"mean": mean, "components": comps, "explained_variance": ev,
            "explained_variance_ratio": ev / total_var.sum(), "singular_values": sv,
            "noise_variance": (total_var.sum() - ev.sum()) / max(min(n, d) - k, 1) if k < min(n, d) else
            torch.zeros((), dtype=torch.float64, device=Z.device), "n": n, "k": k}


def _prep(X: torch.Tensor, prep: dict) -> torch.Tensor:
    Z = (X.double() - prep["scaler"]["mean"]) / prep["scaler"]["scale"]
    if prep.get("pca"):
        Z = (Z - prep["pca"]["mean"]) @ prep["pca"]["components"].T
    return Z


# ------------------------------------------------------------------ logreg
def _logreg(Z: torch.Tensor, y: torch.Tensor, sw: torch.Tensor, C: float) -> dict:
    s = sw.sum()
    wb = torch.zeros(Z.shape[1] + 1, dtype=torch.float64, device=Z.device, requires_grad=True)
    opt = torch.optim.LBFGS([wb], lr=1, max_iter=LBFGS_MAX_ITER, max_eval=LBFGS_MAX_ITER * 2,
                            tolerance_grad=LBFGS_TOL, tolerance_change=0, history_size=10,
                            line_search_fn="strong_wolfe")
    evals = [0]

    def closure():
        opt.zero_grad()
        z = Z @ wb[:-1] + wb[-1]
        loss = (sw * torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")).sum() / s
        loss = loss + (wb[:-1] @ wb[:-1]) / (2 * C * s)
        loss.backward()
        evals[0] += 1
        return loss

    opt.step(closure)
    w = wb.detach()
    return {"coef": w[:-1], "intercept": w[-1], "n_iter": opt.state[opt._params[0]].get("n_iter", 0),
            "n_eval": evals[0]}


# ------------------------------------------------------------------ mlp
def _mlp_init(layers: list[int], rs: np.random.RandomState) -> tuple[list, list]:
    coefs, inters = [], []
    for fan_in, fan_out in zip(layers[:-1], layers[1:]):
        bound = np.sqrt(6.0 / (fan_in + fan_out))  # relu : facteur 6 (_init_coef)
        coefs.append(rs.uniform(-bound, bound, (fan_in, fan_out)))
        inters.append(rs.uniform(-bound, bound, fan_out))
    return coefs, inters


def _mlp_forward(Z: torch.Tensor, coefs: list, inters: list) -> list[torch.Tensor]:
    acts = [Z]
    for i, (W, b) in enumerate(zip(coefs, inters)):
        a = acts[-1] @ W + b
        acts.append(torch.relu(a) if i < len(coefs) - 1 else torch.sigmoid(a))
    return acts


def _mlp(Zf: torch.Tensor, yf: torch.Tensor, wf: torch.Tensor, alpha: float, hidden: list[int],
         Zo: torch.Tensor | None, yo: np.ndarray, go: np.ndarray, max_epochs: int, patience: int) -> dict:
    dev, dt = Zf.device, torch.float32
    rs = np.random.RandomState(0)
    layers = [Zf.shape[1], *hidden, 1]
    c0, i0 = _mlp_init(layers, rs)
    coefs = [torch.tensor(c, dtype=dt, device=dev) for c in c0]
    inters = [torch.tensor(b, dtype=dt, device=dev) for b in i0]
    params = coefs + inters
    ms = [torch.zeros_like(p) for p in params]
    vs = [torch.zeros_like(p) for p in params]
    t = 0
    n = Zf.shape[0]
    order = np.arange(n)
    Zf, yf, wf = Zf.to(dt), yf.to(dt).unsqueeze(1), wf.to(dt)
    best, best_ap, stale, epochs = None, -1.0, 0, 0
    for _ in range(max_epochs):
        # partial_fit : une époque ; scikit-learn recrée RandomState(0) à chaque appel (le premier tire
        # d'abord l'initialisation), puis mélange arange(n) (sklearn.utils.shuffle)
        if epochs:
            rs = np.random.RandomState(0)
        order = np.arange(n)
        rs.shuffle(order)
        idx_all = torch.from_numpy(order).to(dev)
        for start in range(0, n, MLP_BATCH):
            b = idx_all[start:start + MLP_BATCH]
            Xb, yb, wb = Zf[b], yf[b], wf[b]
            acts = _mlp_forward(Xb, coefs, inters)
            sws = wb.sum()
            delta = (acts[-1] - yb) * wb.unsqueeze(1)
            gc, gi = [None] * len(coefs), [None] * len(coefs)
            for li in range(len(coefs) - 1, -1, -1):
                gc[li] = (acts[li].T @ delta + alpha * coefs[li]) / sws
                gi[li] = delta.sum(0) / sws
                if li:
                    delta = (delta @ coefs[li].T) * (acts[li] > 0)
            grads = gc + gi
            t += 1
            lr = ADAM["lr"] * np.sqrt(1 - ADAM["beta_2"] ** t) / (1 - ADAM["beta_1"] ** t)
            for p, g, m, v in zip(params, grads, ms, vs):
                m.mul_(ADAM["beta_1"]).add_(g, alpha=1 - ADAM["beta_1"])
                v.mul_(ADAM["beta_2"]).add_(g * g, alpha=1 - ADAM["beta_2"])
                p.add_(-lr * m / (v.sqrt() + ADAM["eps"]))
        epochs += 1
        if Zo is None:
            continue
        p = _mlp_forward(Zo.to(dt), coefs, inters)[-1].squeeze(1).double().cpu().numpy()
        ap = group_ap(yo, p, go)
        ap = average_precision(yo, p) if ap is None else ap
        if ap > best_ap + 1e-4:
            best = ([c.clone() for c in coefs], [b.clone() for b in inters], epochs, t)
            best_ap, stale = ap, 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best is None:
        best = (coefs, inters, epochs, t)
    return {"coefs": best[0], "intercepts": best[1], "n_iter": best[2], "t": best[3],
            "n_fit": n}


# ------------------------------------------------------------------ entrée
def _np(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    if isinstance(v, dict):
        return {k: _np(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_np(x) for x in v]
    return v


def predict(X: torch.Tensor, fold: dict) -> torch.Tensor:
    """Probabilité de la classe 1 d'un modèle de pli (paramètres en tenseurs)."""
    Z = _prep(X, fold["prep"])
    if "logreg" in fold:
        return torch.sigmoid(Z @ fold["logreg"]["coef"] + fold["logreg"]["intercept"])
    acts = _mlp_forward(Z.float(), fold["mlp"]["coefs"], fold["mlp"]["intercepts"])
    return acts[-1].squeeze(1).double()


def run(X: np.ndarray, y: np.ndarray, plan: dict, device: str | None = None) -> dict:
    """Entraîne un modèle par pli. plan : model, params, et folds, chacun avec fit (positions dans X,
    répétées si rééchantillonnées), w (poids d'images ou None), pred (positions prédites hors pli) ;
    pour le mlp, held (positions des chaînes mises de côté pour l'arrêt) et held_groups, et w déjà
    équilibré par classe comme Lab._fit_mlp. Renvoie probs (hors pli, alignées sur X) et folds
    (paramètres numpy)."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(dev)
    y = np.asarray(y, dtype=np.float64)
    model, params = plan["model"], plan.get("params") or {}
    probs = np.zeros(len(y))
    folds = []
    for f in plan["folds"]:
        fit = torch.as_tensor(np.asarray(f["fit"], dtype=np.int64), device=dev)
        Xf = Xt[fit]
        prep = {"scaler": _scaler(Xf)}
        Z = (Xf.double() - prep["scaler"]["mean"]) / prep["scaler"]["scale"]
        del Xf
        if params.get("pca"):
            prep["pca"] = _pca(Z, int(params["pca"]))
            Z = (Z - prep["pca"]["mean"]) @ prep["pca"]["components"].T
        yf = torch.as_tensor(y[np.asarray(f["fit"])], device=dev)
        w = torch.as_tensor(np.ones(len(f["fit"])) if f.get("w") is None else np.asarray(f["w"], dtype=np.float64),
                            device=dev)
        fold = {"prep": prep}
        if model == "logreg":
            # class_weight="balanced" : n / (2 x effectif de la classe), sur les lignes rééchantillonnées
            cnt = torch.stack([(yf == 0).sum(), (yf == 1).sum()]).double()
            cw = len(yf) / (2 * cnt.clamp_min(1))
            sw = w * torch.where(yf == 1, cw[1], cw[0])
            fold["logreg"] = _logreg(Z, yf.double(), sw, float(params.get("C", 0.1)))
        else:
            held = np.asarray(f.get("held", []), dtype=np.int64)
            Zo = _prep(Xt[torch.as_tensor(held, device=dev)], prep) if len(held) else None
            fold["mlp"] = _mlp(Z, yf, w, float(params.get("alpha", plan["mlp_alpha"])),
                               list(params.get("hidden", [256])), Zo, y[held] if len(held) else None,
                               np.asarray(f.get("held_groups", [])), plan["mlp_max_epochs"],
                               plan["mlp_patience"])
        del Z
        pred = np.asarray(f["pred"], dtype=np.int64)
        probs[pred] = predict(Xt[torch.as_tensor(pred, device=dev)], fold).cpu().numpy()
        folds.append(_np(fold))
    return {"probs": probs, "folds": folds, "device": str(dev)}

import numpy as np
import pandas as pd
import torch
import umap
import hdbscan
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from collections import Counter
import re
import spacy
from torch.utils.data import DataLoader


# ---------------------------------------------------------
# Global stopwords
# ---------------------------------------------------------
nlp = spacy.load("en_core_web_sm")
spacy_stopwords = nlp.Defaults.stop_words


# ============================================================================
#                              EDGE CASE ANALYZER
# ============================================================================
class EdgeCaseAnalyzer:

    # ---------------------------------------------------------
    # Constructor — uses IN-MEMORY model + tokenizer
    # ---------------------------------------------------------
    def __init__(
        self,
        df,
        model,
        tokenizer,
        text_column="text",
        batch_size=32,
        max_length=256,
    ):
        """
        df          : misclassified dataframe
        model       : finetuned HF model already in memory
        tokenizer   : tokenizer used during training
        """

        self.df = df.copy()
        self.texts = df[text_column].tolist()

        self.model = model
        self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.max_length = max_length

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Placeholders
        self.embeddings = None
        self.coords = None
        self.labels = None
        self.cluster_keywords = None

        # Auto-run embedding computation
        self._compute_embeddings()


    # ---------------------------------------------------------
    # Compute embeddings using CLS token
    # ---------------------------------------------------------
    def _compute_embeddings(self):
        print("🔍 Computing embeddings from in-memory model…")

        all_embs = []
        texts = self.texts

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            tokens.pop("token_type_ids", None)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            with torch.no_grad():
                # ⭐ CALL THE ENCODER DIRECTLY INSTEAD OF model()
                for name, module in self.model.named_children():
                    print("Child:", name, "→", type(module))
                encoder_out = self.model.transformer(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                return_dict=True
                )
                

            # extract CLS embedding
            cls_emb = encoder_out.last_hidden_state[:, 0, :].detach().cpu().numpy()
            all_embs.append(cls_emb)

        self.embeddings = np.vstack(all_embs)
        print(f"✔ Embeddings computed: shape = {self.embeddings.shape}")




    # ---------------------------------------------------------
    # Extract keyword labels for each cluster
    # ---------------------------------------------------------
    def auto_label_clusters(self, top_k=8):

        labels = self.labels
        texts = self.texts
        cluster_labels = {}

        for cid in sorted(set(labels)):
            if cid == -1:
                continue

            cluster_texts = [t for t, l in zip(texts, labels) if l == cid]

            words = []
            for t in cluster_texts:
                w = re.findall(r"[A-Za-z]+", t.lower())
                words.extend(w)

            words = [
                w for w in words
                if w not in spacy_stopwords and len(w) > 2
            ]

            if not words:
                cluster_labels[cid] = "(no keywords)"
            else:
                top_words = [w for w, c in Counter(words).most_common(top_k)]
                cluster_labels[cid] = ", ".join(top_words)

        self.cluster_keywords = cluster_labels
        return cluster_labels



    # ---------------------------------------------------------
    # Run UMAP + HDBSCAN clustering
    # ---------------------------------------------------------
    def run_clustering(self, min_cluster_size=10, n_neighbors=20, min_dist=0.05):

        normed = StandardScaler().fit_transform(self.embeddings)

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(normed)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(coords)

        self.coords = coords
        self.labels = labels

        return labels, coords



    # ---------------------------------------------------------
    # Distance-to-centroid score with noise fallback
    # ---------------------------------------------------------
    def compute_distance_scores(self):

        coords = self.coords
        labels = self.labels

        unique = [cid for cid in set(labels) if cid != -1]

        # SAFETY FIX: no clusters found
        if len(unique) == 0:
            print("⚠️ HDBSCAN found NO clusters — treating everything as noise.")
            dist = np.ones(len(coords))
            self.df["dist_score"] = dist
            return dist

        # compute centroids
        centroids = np.array([
            coords[labels == cid].mean(axis=0)
            for cid in unique
        ])

        # ensure correct shape
        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)

        _, dist = pairwise_distances_argmin_min(coords, centroids)
        self.df["dist_score"] = dist
        return dist



    # ---------------------------------------------------------
    # Structural anomaly score
    # ---------------------------------------------------------
    def compute_structural_scores(self):

        if not all(col in self.df.columns
                   for col in ["token_count", "max_dep_depth", "clause_count"]):

            self.df["token_count"] = [len(t.split()) for t in self.texts]
            self.df["max_dep_depth"] = self.df["token_count"]
            self.df["clause_count"] = 1

        structural_cols = ["token_count", "max_dep_depth", "clause_count"]

        structural = self.df[structural_cols].rank(pct=True).mean(axis=1)
        self.df["struct_score"] = structural

        return structural



    # ---------------------------------------------------------
    # Probability-aware edge-case score
    # ---------------------------------------------------------
    def compute_edge_scores(self):

        dist = self.compute_distance_scores()
        struct = self.compute_structural_scores()

        df = self.df

        low_conf = np.zeros(len(df))
        wrong_conf = np.zeros(len(df))
        true_conf_score = np.zeros(len(df))

        if "predicted_prob" in df.columns and "true_prob" in df.columns:
            pred_conf = df["predicted_prob"]
            true_conf = df["true_prob"]

            low_conf = 1 - pred_conf
            wrong_conf = pred_conf
            true_conf_score = 1 - true_conf

        noise = (self.labels == -1).astype(float)

        combined = (
              (dist / dist.max()) * 0.30
            + struct * 0.20
            + noise * 0.30
            + low_conf * 0.15
            + wrong_conf * 0.15
            + true_conf_score * 0.10
        )

        self.df["edge_score"] = combined
        return combined



    # ---------------------------------------------------------
    # Top N edge cases (now supports n=None => return all)
    # ---------------------------------------------------------
    def get_top_edge_cases(self, n=None):

        # ensure scores exist
        if "edge_score" not in self.df.columns:
            self.compute_edge_scores()

        sorted_df = self.df.sort_values("edge_score", ascending=False)

        # n is None => return all rows
        if n is None:
            return sorted_df
        return sorted_df.head(n)



    # ---------------------------------------------------------
    # Export edge cases table (all or top-n) to CSV or return df
    # ---------------------------------------------------------
    def export_edge_cases(self, filepath=None, n=None):
        """
        filepath : if provided, writes CSV to path
        n        : if None -> export all, otherwise export top-n
        """
        df_out = self.get_top_edge_cases(n=n)

        if filepath:
            df_out.to_csv(filepath, index=False)
        return df_out


    # ---------------------------------------------------------
    # Tiny semantic outlier clusters
    # ---------------------------------------------------------
    def get_tiny_clusters(self, min_size=20):

        counts = Counter(self.labels)
        return [cid for cid, c in counts.items() if 0 < cid != -1 and c < min_size]



    # ---------------------------------------------------------
    # Representative examples
    # ---------------------------------------------------------
    def get_cluster_centroid_texts(self, cid, k=5):

        coords = self.coords
        labels = self.labels

        pts = coords[labels == cid]
        center = pts.mean(axis=0)

        dists = np.linalg.norm(pts - center, axis=1)
        idx = np.argsort(dists)[:k]

        return [self.texts[i] for i in np.where(labels == cid)[0][idx]]



    # ---------------------------------------------------------
    # Plotting — figure returned for W&B logging
    # ---------------------------------------------------------
    def plot_clusters(self, title="Edge Case Clusters"):

        if self.coords is None or self.labels is None:
            raise RuntimeError("Run clustering first.")

        fig, ax = plt.subplots(figsize=(10, 8))

        sc = ax.scatter(
            self.coords[:, 0],
            self.coords[:, 1],
            c=self.labels,
            cmap="Spectral",
            alpha=0.85,
            s=10
        )

        ax.set_title(title)
        plt.colorbar(sc, ax=ax)

        return fig



    # ---------------------------------------------------------
    # FULL PIPELINE WRAPPER
    # ---------------------------------------------------------
    def run_full_analysis(self, top_n=None, tiny_threshold=20, title="Edge Case Clusters"):

        # Step 1: clustering
        self.run_clustering()

        # Step 2: label clusters
        self.auto_label_clusters()

        # Step 3: compute probability + structure + distance scoring
        self.compute_edge_scores()

        # Step 4: retrieve edge cases
        top_edges = self.get_top_edge_cases(top_n)

        # Step 5: tiny clusters identification
        tiny = self.get_tiny_clusters(tiny_threshold)

        # Step 6: get figure to log to wandb
        fig = self.plot_clusters(title)

        return {
            "top_edge_cases": top_edges,
            "tiny_clusters": tiny,
            "cluster_keywords": self.cluster_keywords,
            "coords": self.coords,
            "labels": self.labels,
            "plot": fig,
        }

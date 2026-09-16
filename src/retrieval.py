"""
retrieval.py — find the brand's most similar PAST RESOLVED threads for a new
message, so replies can be grounded in how the brand actually resolved things.

Two backends, chosen automatically:
  embed : sentence-transformers MiniLM embeddings (semantic, better) — used when
          the model is available / downloadable.
  tfidf : TF-IDF cosine (lexical) — offline fallback, no download, works anywhere.

The agent grounds replies ONLY on retrieved resolutions; if the best match is
too weak (below a similarity floor), the agent escalates instead of guessing.
"""
import pickle

import numpy as np

EMBED_MODEL = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self, backend="auto"):
        self.backend = backend
        self.meta = []          # [{thread_id, customer_msg, resolution}]
        self._matrix = None     # (n,d) float32 for embed, or sparse for tfidf
        self._vec = None        # TfidfVectorizer (tfidf)
        self._model = None      # SentenceTransformer (embed)
        self._mode = None       # 'embed' | 'tfidf'

    def _load_embed_model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBED_MODEL)

    def build(self, corpus):
        self.meta = corpus
        texts = [c["customer_msg"] for c in corpus]
        if self.backend in ("auto", "embed"):
            try:
                self._model = self._load_embed_model()
                emb = self._model.encode(texts, normalize_embeddings=True,
                                         batch_size=64, show_progress_bar=True)
                self._matrix = np.asarray(emb, dtype="float32")
                self._mode = "embed"
                print(f"[retriever] built EMBED index over {len(texts)} threads")
                return
            except Exception as e:
                if self.backend == "embed":
                    raise
                print(f"[retriever] embeddings unavailable -> TF-IDF fallback ({str(e)[:60]})")
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self._matrix = self._vec.fit_transform(texts)
        self._mode = "tfidf"
        print(f"[retriever] built TFIDF index over {len(texts)} threads")

    def query(self, text, k=3):
        if self._mode == "embed":
            q = self._model.encode([text], normalize_embeddings=True)[0].astype("float32")
            sims = self._matrix @ q
        else:
            from sklearn.metrics.pairwise import linear_kernel
            q = self._vec.transform([text])
            sims = linear_kernel(q, self._matrix).ravel()
        idx = np.argsort(-sims)[:k]
        return [{**self.meta[i], "score": float(sims[i])} for i in idx]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"mode": self._mode, "meta": self.meta,
                         "matrix": self._matrix, "vec": self._vec}, f)
        print(f"[retriever] saved -> {path}")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        r = cls()
        r._mode = d["mode"]; r.meta = d["meta"]
        r._matrix = d["matrix"]; r._vec = d["vec"]
        if r._mode == "embed":
            r._model = r._load_embed_model()
        return r

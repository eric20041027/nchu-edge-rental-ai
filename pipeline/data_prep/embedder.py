"""Precompute property embeddings for vector search."""
import pickle
from typing import Optional

import pandas as pd

from .base import BaseProcessor
from .config import DataPrepConfig
from .models import PropertyEmbedding, EmbeddingBatch


class EmbeddingPrecomputer(BaseProcessor):
    """Precomputes semantic embeddings for rental properties.

    Uses a sentence embedding model (default: paraphrase-MiniLM-L6-v2) to
    create fixed-size dense vectors for each property description, enabling
    efficient vector search and semantic matching.
    """

    def __init__(self, config: DataPrepConfig):
        super().__init__(config)
        self.model_name = config.embedding_model
        self.batch_size = config.embedding_batch_size
        self.device = config.embedding_device

    def run(self, properties_df: Optional[pd.DataFrame] = None) -> EmbeddingBatch:
        """Precompute embeddings for properties.

        Args:
            properties_df: Optional DataFrame with property data. If None, loads from config.

        Returns:
            EmbeddingBatch containing all property embeddings
        """
        # Load data
        if properties_df is None:
            self.log_step(f"Loading merged data from {self.config.merged_csv}")
            properties_df = pd.read_csv(self.config.merged_csv)
        else:
            self.log_step(f"Using provided data ({len(properties_df)} rows)")

        self.log_result("Total properties", len(properties_df))

        # Lazy import (only if embeddings are needed)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self.logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            return EmbeddingBatch(embeddings=[], count=0, model_name=self.model_name)

        # Load model
        self.log_step(f"Loading embedding model: {self.model_name}")
        model = SentenceTransformer(self.model_name, device=self.device)

        # Create property texts
        self.log_step("Creating property text representations")
        texts = []
        property_ids = []

        for _, row in properties_df.iterrows():
            url = row.get("網址", "")
            address = row.get("地址", "")
            room_type = row.get("類型", "")
            rent = row.get("租金", "")
            furniture = row.get("家具設施", "")

            # Concatenate key fields into a single text representation
            text = f"{address} {room_type} {rent} {furniture}"
            texts.append(text)
            property_ids.append(url)

        # Compute embeddings in batches
        self.log_step(f"Computing embeddings ({self.model_name})")
        embeddings_list = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_ids = property_ids[i : i + self.batch_size]

            embeddings = model.encode(batch_texts, batch_size=self.batch_size, convert_to_numpy=True)

            for emb_idx, emb in enumerate(embeddings):
                embedding_obj = PropertyEmbedding(
                    property_id=batch_ids[emb_idx],
                    text=batch_texts[emb_idx],
                    embedding=emb.tolist(),
                    model_name=self.model_name
                )
                embeddings_list.append(embedding_obj)

            if (i + self.batch_size) % 1000 == 0 or i + self.batch_size >= len(texts):
                self.log_result(f"Processed", f"{min(i + self.batch_size, len(texts))}/{len(texts)}")

        # Create batch
        batch = EmbeddingBatch(
            embeddings=embeddings_list,
            count=len(embeddings_list),
            model_name=self.model_name
        )

        # Save
        self.log_step(f"Saving embeddings to {self.config.embeddings_pkl}")
        with open(self.config.embeddings_pkl, "wb") as f:
            pickle.dump(batch, f)

        self.log_result("Embeddings saved", len(embeddings_list))

        return batch

    def load_embeddings(self) -> EmbeddingBatch:
        """Load precomputed embeddings from disk."""
        self.log_step(f"Loading embeddings from {self.config.embeddings_pkl}")
        with open(self.config.embeddings_pkl, "rb") as f:
            batch = pickle.load(f)
        self.log_result("Embeddings loaded", batch.count)
        return batch

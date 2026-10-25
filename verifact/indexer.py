import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


class ClinicalVectorIndexer:
    """
    Indexes raw clinical progress notes and discharge summaries into a local vector store (Qdrant).
    Uses sliding window chunking with overlap to preserve clinical context across sentence boundaries,
    and embeds chunks using local sentence-transformers (all-MiniLM-L6-v2).
    """

    def __init__(
        self,
        collection_name: str = "clinical_notes",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        client: Optional[QdrantClient] = None
    ):
        self.collection_name = collection_name
        self.client = client or QdrantClient(":memory:")
        self.encoder = SentenceTransformer(embedding_model_name)
        self.vector_dim = self.encoder.get_sentence_embedding_dimension()

        # Initialize collection
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.vector_dim, distance=models.Distance.COSINE),
        )

    def chunk_text(self, text: str, chunk_size: int = 150, overlap: int = 30) -> List[Dict[str, Any]]:
        """
        Splits text into overlapping chunks of words while retaining token boundaries.
        Returns list of dicts with text and metadata.
        """
        words = text.split()
        if not words:
            return []

        chunks = []
        i = 0
        chunk_idx = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_str = " ".join(chunk_words)
            chunks.append({
                "chunk_index": chunk_idx,
                "text": chunk_str,
                "start_word_idx": i,
                "end_word_idx": min(i + chunk_size, len(words))
            })
            chunk_idx += 1
            if i + chunk_size >= len(words):
                break
            i += max(1, (chunk_size - overlap))

        return chunks

    def index_note(self, note_id: str, note_text: str) -> int:
        """
        Chunks and encodes a clinical note, then upserts the points into Qdrant.
        Returns the number of indexed chunks.
        """
        chunks = self.chunk_text(note_text)
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        points = []

        for c, emb in zip(chunks, embeddings):
            chunk_idx = c["chunk_index"]
            point_id = abs(hash(f"{note_id}_{chunk_idx}")) % (10**9)
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=emb.tolist(),
                    payload={
                        "note_id": note_id,
                        "chunk_index": chunk_idx,
                        "text": c["text"],
                        "start_word_idx": c["start_word_idx"],
                        "end_word_idx": c["end_word_idx"],
                    }
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def retrieve_chunks(self, query: str, note_id: Optional[str] = None, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieves top_k premise chunks closest to the query embedding.
        Optionally filters by note_id.
        """
        query_vector = self.encoder.encode(query, convert_to_numpy=True).tolist()
        
        query_filter = None
        if note_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(key="note_id", match=models.MatchValue(value=note_id))
                ]
            )

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter
        )

        results = []
        for point in search_result.points:
            results.append({
                "chunk_id": point.id,
                "score": point.score,
                "text": point.payload.get("text", ""),
                "note_id": point.payload.get("note_id", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
            })
        return results

    # Backward compatibility alias
    def retrieve_evidence(self, query: str, note_id: str, top_k: int = 2) -> List[str]:
        hits = self.retrieve_chunks(query=query, note_id=note_id, top_k=top_k)
        return [hit["text"] for hit in hits]


# Alias for backward compatibility with existing scripts
ClinicalNoteIndexer = ClinicalVectorIndexer
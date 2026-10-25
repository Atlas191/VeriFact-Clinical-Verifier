from typing import List, Dict, Any, Optional
from verifact.indexer import ClinicalVectorIndexer


class PremiseRetriever:
    """
    Modular Premise Retrieval Engine.
    Queries the vector index using decomposed atomic claims to retrieve the top-k
    most relevant premise chunks from the source clinical note, preventing context
    dilution and hallucination during downstream LLM inference.
    """

    def __init__(self, indexer: Optional[ClinicalVectorIndexer] = None):
        self.indexer = indexer or ClinicalVectorIndexer()

    def index_note(self, note_id: str, note_text: str) -> int:
        """Indexes note into underlying vector store."""
        return self.indexer.index_note(note_id=note_id, note_text=note_text)

    def retrieve_premise_for_claim(
        self,
        claim_text: str,
        note_id: str,
        top_k: int = 2,
        join_delimiter: str = " "
    ) -> Dict[str, Any]:
        """
        Retrieves top-k relevant premise chunks for a single claim and returns
        both the raw hits and concatenated context.
        """
        chunks = self.indexer.retrieve_chunks(query=claim_text, note_id=note_id, top_k=top_k)
        if not chunks:
            return {
                "claim": claim_text,
                "note_id": note_id,
                "premise_text": "",
                "chunks": [],
                "top_score": 0.0
            }

        premise_text = join_delimiter.join([c["text"] for c in chunks])
        top_score = max([c["score"] for c in chunks]) if chunks else 0.0

        return {
            "claim": claim_text,
            "note_id": note_id,
            "premise_text": premise_text,
            "chunks": chunks,
            "top_score": top_score
        }

    def retrieve_premises_for_claims(
        self,
        claims: List[str],
        note_id: str,
        top_k: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Batch retrieval of premise chunks for multiple atomic claims against a note.
        """
        results = []
        for claim in claims:
            res = self.retrieve_premise_for_claim(claim_text=claim, note_id=note_id, top_k=top_k)
            results.append(res)
        return results

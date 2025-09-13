import os
from topk_sdk import Client
from topk_sdk.schema import text, f32_vector, vector_index, keyword_index, int
from topk_sdk.query import select, field, fn


class TopKClient:
    def __init__(self, region=None, api_key=None):
        self.region = region or os.getenv("TOPK_REGION", "aws-us-east-1-elastica")
        self.api_key = api_key or os.getenv("TOPK_API_KEY", None)
        self.collection = os.getenv("TOPK_COLLECTION", "benchmark")
        
        if not self.api_key:
            raise ValueError("TOPK_API_KEY environment variable is required but not set. Please set it with: export TOPK_API_KEY='your_api_key_here'")
        
        self.client = Client(api_key=self.api_key, region=self.region)

    def setup(self, dim: int = 384):
        # Create collection with schema matching Milvus fields, use provided dimension
        # Try creating collection without vector index first, then add index
        schema = {
            "id": int().required(),
            "vector": f32_vector(dimension=dim).required(),
            "track": text().required(),
            "artist": text().required(),
            "genre": text(),
            "seeds": text(),
            "text": text(),
        }
        try:
            # Create collection first
            self.client.collections().create(self.collection, schema=schema)
            print(f"Collection {self.collection} created successfully")
            
            # Try to add vector index separately
            try:
                self.client.collections().create_index(
                    self.collection, 
                    "vector", 
                    vector_index(metric="cosine")
                )
                print("Vector index created successfully")
            except Exception as idx_e:
                print(f"Vector index creation failed: {idx_e}")
                # Continue anyway - maybe the index is not needed for basic operations
                
        except Exception as e:
            if "already exists" not in str(e):
                print(f"Collection creation error: {e}")
                # Try to continue anyway

    def upsert(self, vectors, payloads):
        # Upsert documents with all fields for parity, using correct TopK SDK upsert
        docs = []
        for i, (vec, meta) in enumerate(zip(vectors, payloads)):
            row_id = meta.get("row_id", i)
            doc = {
                "_id": str(row_id),
                "id": row_id,
                "vector": vec,
                "track": meta.get("track", "unknown"),
                "artist": meta.get("artist", "unknown"),
                "genre": meta.get("genre", ""),
                "seeds": meta.get("seeds", ""),
                "text": meta.get("text", ""),
            }
            docs.append(doc)
        BATCH_SIZE = 200
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i : i + BATCH_SIZE]
            self.client.collection(self.collection).upsert(batch)

    def search(self, vector, top_k=10):
        # Vector search using TopK SDK query API
        col = self.client.collection(self.collection)
        docs = col.query(
            select(
                "id",
                "track",
                "artist",
                "genre",
                "seeds",
                "text",
                vector_similarity=fn.vector_distance("vector", vector),
            ).topk(field("vector_similarity"), top_k, asc=False)
        )
        # Return in the same format as other clients
        out = []
        for d in docs:
            out.append(
                {
                    "id": d.get("id"),
                    "score": d.get("vector_similarity", 0.0),
                    "payload": {
                        "row_id": d.get("id"),
                        "track": d.get("track"),
                        "artist": d.get("artist"),
                        "genre": d.get("genre"),
                        "seeds": d.get("seeds"),
                        "text": d.get("text"),
                    },
                }
            )
        return out

    def teardown(self):
        # Optionally delete collection (not required for benchmarking)
        pass

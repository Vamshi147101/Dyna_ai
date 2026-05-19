# test_retrieval_v2.py
import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "./K-files/Dyna_ai/chroma_db_v2"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-base-en-v1.5"
)

def search(query, collection_name="keyword_manual", n=5, filter_card=None, filter_section=None):
    """
    Search with optional metadata filtering.
    """
    collection = chroma_client.get_collection(
        name=collection_name, embedding_function=embedding_fn
    )
    
    # Build filter
    where_filter = {}
    if filter_card:
        where_filter["primary_card"] = filter_card
    if filter_section:
        where_filter["section_type"] = filter_section
    
    results = collection.query(
        query_texts=[query],
        n_results=n,
        where=where_filter if where_filter else None
    )
    
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    if filter_card:
        print(f"Filter (card): {filter_card}")
    if filter_section:
        print(f"Filter (section): {filter_section}")
    print(f"{'='*70}")
    
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        score = 1 - results['distances'][0][i]
        
        print(f"\n--- #{i+1} (score: {score:.3f}) ---")
        print(f"Card: {meta.get('primary_card', 'N/A')}")
        print(f"Type: {meta.get('section_type', 'N/A')}")
        print(f"Page: {meta.get('page', 'N/A')}")
        print(f"Preview: {doc[:300]}...")

if __name__ == "__main__":
    # Direct card lookup
    search("*MAT_ELASTIC fields", filter_card="*MAT_ELASTIC")
    
    # Get only field definitions
    search("contact parameters", filter_section="field_table")
    
    # General search
    search("Johnson Cook failure strain", "material_manual")
    
    # Example manual
    search("beam impact", "example_manual")
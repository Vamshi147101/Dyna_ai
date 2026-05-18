# build_vectordb.py
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import re

# ─── CONFIGURATION ───
CHROMA_PATH = "./chroma_db"
EXTRACTED_DIR = "D:\\vamshi\\ML\\K-files\\Dyna_ai\\extracted_manuals"
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 2500

# ─── INITIALIZE ───
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collections = {
    "keyword_manual": chroma_client.get_or_create_collection(
        name="keyword_manual",
        embedding_function=embedding_fn,
        metadata={"description": "LS-DYNA Keyword Manual - card definitions and parameters"}
    ),
    "material_manual": chroma_client.get_or_create_collection(
        name="material_manual",
        embedding_function=embedding_fn,
        metadata={"description": "LS-DYNA Material Manual - material model documentation"}
    ),
    "example_manual": chroma_client.get_or_create_collection(
        name="example_manual",
        embedding_function=embedding_fn,
        metadata={"description": "LS-DYNA Example Manual - worked examples"}
    ),
    "theory_manual": chroma_client.get_or_create_collection(
        name="theory_manual",
        embedding_function=embedding_fn,
        metadata={"description": "LS-DYNA Theory Manual - mathematical background"}
    ),
    "multiphysics_manual": chroma_client.get_or_create_collection(
        name="multiphysics_manual",
        embedding_function=embedding_fn,
        metadata={"description": "LS-DYNA Multiphysics Manual"}
    ),
}

# ─── IMPROVED CHUNKING ───
def chunk_by_semantic_units(text):
    """
    Chunks text preserving card definitions and section structures.
    """
    # Split by card boundaries (*CARD_NAME)
    card_splits = re.split(r'\n(?=\*[A-Z][A-Z_0-9]+)', text)
    
    all_chunks = []
    
    for section in card_splits:
        section = section.strip()
        if not section:
            continue
        
        # Small sections stay whole
        if len(section) <= MAX_CHUNK_SIZE:
            all_chunks.append(section)
            continue
        
        # Split large sections by sub-headings
        subsections = re.split(
            r'\n(?=(?:Purpose|Remarks?|Definition|Parameters?|Fields?'
            r'|Card \d+|Input|Output|Description|Example|Usage'
            r'|Available cards|VARIABLE|DESCRIPTION|FORMAT'
            r'|Notes|Warning|Restrictions|Guidelines)\b)',
            section,
            flags=re.IGNORECASE
        )
        
        for sub in subsections:
            sub = sub.strip()
            if not sub:
                continue
            
            if len(sub) <= MAX_CHUNK_SIZE:
                all_chunks.append(sub)
            else:
                # Final fallback: paragraph-level chunking
                paragraphs = sub.split('\n\n')
                current_chunk = ""
                
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    
                    if len(current_chunk) + len(para) > MAX_CHUNK_SIZE and current_chunk:
                        all_chunks.append(current_chunk.strip())
                        # Keep first line as context overlap
                        first_line = current_chunk.split('\n')[0]
                        current_chunk = first_line + '\n' + para if first_line else para
                    else:
                        current_chunk = '\n\n'.join([current_chunk, para]) if current_chunk else para
                
                if current_chunk.strip():
                    all_chunks.append(current_chunk.strip())
    
    # Filter by size
    return [c for c in all_chunks if MIN_CHUNK_SIZE <= len(c) <= MAX_CHUNK_SIZE * 2]

# ─── METADATA EXTRACTION ───
def extract_metadata_from_chunk(chunk_text, manual_type, page_num):
    """Extract rich metadata for better retrieval accuracy"""
    metadata = {
        'manual_type': manual_type,
        'page': page_num,
        'chunk_size': len(chunk_text),
        'word_count': len(chunk_text.split())
    }
    
    # Primary card name (first *CARD found)
    card_match = re.search(r'\*([A-Z][A-Z_0-9]+)', chunk_text)
    if card_match:
        metadata['primary_card'] = f"*{card_match.group(1)}"
    
    # All cards referenced
    all_cards = re.findall(r'\*([A-Z][A-Z_0-9]+)', chunk_text)
    if all_cards:
        metadata['all_cards'] = ','.join(list(set(all_cards))[:25])
    
    # Section type
    if re.search(r'\bPurpose\b', chunk_text, re.IGNORECASE):
        metadata['section_type'] = 'purpose'
    elif re.search(r'\b(Fields?|Variables?|Card \d)\b', chunk_text, re.IGNORECASE):
        metadata['section_type'] = 'field_definition'
    elif re.search(r'\bRemarks?\b', chunk_text, re.IGNORECASE):
        metadata['section_type'] = 'remarks'
    elif re.search(r'\bExample\b', chunk_text, re.IGNORECASE):
        metadata['section_type'] = 'example'
    else:
        metadata['section_type'] = 'general'
    
    # Is material model?
    if manual_type == 'material_manual' or re.search(r'\*MAT_', chunk_text):
        metadata['is_material'] = 'true'
    
    return metadata

# ─── COLLECTION ROUTING ───
def get_collection_for_file(filename):
    name_lower = filename.lower()
    if 'material' in name_lower:
        return 'material_manual'
    elif 'example' in name_lower:
        return 'example_manual'
    elif 'theory' in name_lower:
        return 'theory_manual'
    elif 'multiphysics' in name_lower or 'multi' in name_lower:
        return 'multiphysics_manual'
    elif 'keyword' in name_lower:
        return 'keyword_manual'
    return 'keyword_manual'

# ─── MAIN INDEXING ───
def index_manuals():
    extracted_dir = Path(EXTRACTED_DIR)
    txt_files = list(extracted_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"No extracted .txt files found in {EXTRACTED_DIR}")
        print("Run extract_pdfs.py first")
        return
    
    print(f"Found {len(txt_files)} text files to index\n")
    total_chunks = 0
    
    for txt_file in txt_files:
        collection_name = get_collection_for_file(txt_file.name)
        collection = collections[collection_name]
        
        print(f"Indexing: {txt_file.name} → {collection_name}")
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by pages
        pages = re.split(r'--- PAGE (\d+) ---', content)
        
        chunk_count = 0
        
        # Process in pairs: page_number, page_text
        for i in range(1, len(pages), 2):
            try:
                page_num = int(pages[i])
                page_text = pages[i + 1] if i + 1 < len(pages) else ""
            except (ValueError, IndexError):
                continue
            
            if not page_text.strip():
                continue
            
            # Use semantic chunking
            chunks = chunk_by_semantic_units(page_text)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{txt_file.stem}_p{page_num}_c{chunk_idx}"
                
                metadata = extract_metadata_from_chunk(
                    chunk, collection_name, page_num
                )
                metadata['source_file'] = txt_file.name
                metadata['chunk_index'] = chunk_idx
                
                collection.add(
                    documents=[chunk],
                    metadatas=[metadata],
                    ids=[chunk_id]
                )
                
                chunk_count += 1
        
        print(f"  → {chunk_count} chunks indexed")
        total_chunks += chunk_count
    
    print(f"\n{'='*50}")
    print(f"Indexing complete! Total chunks: {total_chunks}")
    for name, coll in collections.items():
        print(f"  {name}: {coll.count()} chunks")

if __name__ == "__main__":
    index_manuals()
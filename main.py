import os
from dotenv import load_dotenv
import glob
import tiktoken
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# =====================================================================
# 1. INITIALISATION & CONFIGURATION
# =====================================================================
# Set your OpenAI API Key (or rely on system environment variables)
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize Tokenizer for gpt-4o-mini optimization tracking
TOKENIZER = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text: str) -> int:
    """Helper function to keep precise track of token counts."""
    return len(TOKENIZER.encode(text))

# =====================================================================
# 2. INTENT-DRIVEN GEOLOGICAL SEMANTIC CHUNKING
# =====================================================================
def semantic_chunk_file(file_path: str) -> List[Document]:
    """
    Reads a drill log file and applies semantic chunking based on 
    the statistical differences in semantic embeddings across sentences.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # Initialize the semantic chunker using OpenAI Embeddings
    # Uses a percentile threshold to split chunks dynamically when context shifts
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    chunker = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile"
    )
    
    # Wrap text in a LangChain Document format and chunk
    base_doc = Document(page_content=raw_text, metadata={"source": os.path.basename(file_path)})
    semantic_docs = chunker.split_documents([base_doc])
    
    return semantic_docs

# =====================================================================
# 3. DYNAMIC SLIDING WINDOW (TOKEN-OPTIMISED BATCHING)
# =====================================================================
def build_dynamic_sliding_windows(semantic_docs: List[Document], max_window_tokens: int = 1500, overlap_tokens: int = 250) -> List[Dict[str, Any]]:
    """
    Bundles semantic chunks into an optimized sliding window to prevent LLM fatigue
    and ensure geological boundaries aren't aggressively truncated across prompts.
    """
    windows = []
    current_window_docs = []
    current_window_tokens = 0
    
    for doc in semantic_docs:
        doc_tokens = count_tokens(doc.page_content)
        
        # If a single semantic chunk is exceptionally long, handle separately
        if doc_tokens > max_window_tokens:
            windows.append({
                "source": doc.metadata["source"],
                "content": doc.page_content,
                "token_count": doc_tokens
            })
            continue
            
        if current_window_tokens + doc_tokens > max_window_tokens:
            # Consolidate the active window text before slicing forward
            window_text = "\n\n--- [Semantic Context Shift] ---\n\n".join([d.page_content for d in current_window_docs])
            windows.append({
                "source": doc.metadata["source"],
                "content": window_text,
                "token_count": current_window_tokens
            })
            
            # Slide window back by applying overlap to retain continuity
            overlap_docs = []
            accumulated_overlap_tokens = 0
            for d in reversed(current_window_docs):
                t = count_tokens(d.page_content)
                if accumulated_overlap_tokens + t <= overlap_tokens:
                    overlap_docs.insert(0, d)
                    accumulated_overlap_tokens += t
                else:
                    break
            
            current_window_docs = overlap_docs + [doc]
            current_window_tokens = accumulated_overlap_tokens + doc_tokens
        else:
            current_window_docs.append(doc)
            current_window_tokens += doc_tokens
            
    # Flush remaining documents
    if current_window_docs:
        window_text = "\n\n--- [Semantic Context Shift] ---\n\n".join([d.page_content for d in current_window_docs])
        windows.append({
            "source": semantic_docs[0].metadata["source"] if semantic_docs else "Unknown",
            "content": window_text,
            "token_count": current_window_tokens
        })
        
    return windows

# =====================================================================
# 4. STRUCTURED ANALYSIS SCHEMA & AGENT LLM EXECUTION
# =====================================================================
class LithologyExtraction(BaseModel):
    hole_id: str = Field(description="The unique identifier or borehole code extracted from log header.")
    rock_type: str = Field(description="Dominant rock type or soil composition identified.")
    texture: str = Field(description="Rock matrix texture profile (e.g., foliated, aphanitic, clastic).")
    grain_size: str = Field(description="Grain classification range observed.")
    key_anomalies: List[str] = Field(description="Notable geological structural observations, mineralization, or fractures.")

def execute_drill_log_scan(window_payload: Dict[str, Any], model_name: str = "gpt-4o-mini"):
    """
    Executes structural analysis extraction using a token-frugal prompt
    and schema enforcing json output parser.
    """
    # Instruct model via an optimized structural system template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert production geologist mining parser. Analyse the following raw drill log window segment. "
                   "Extract the core structural geological attributes cleanly into JSON format according to schema instructions. "
                   "If metadata fields or classifications are missing from this specific window block, label as 'Context Missing'."),
        ("human", "Log Segment Source: {source}\n\nContent:\n{content}")
    ])
    
    # Define a cheap, fast, deterministic extraction run
    llm = ChatOpenAI(model=model_name, temperature=0.0)
    output_parser = JsonOutputParser(pydantic_object=LithologyExtraction)
    
    # Chain components together
    chain = prompt | llm | output_parser
    
    try:
        response = chain.invoke({
            "source": window_payload["source"],
            "content": window_payload["content"]
        })
        return response
    except Exception as e:
        return {"error": f"Failed extraction window: {str(e)}"}

# =====================================================================
# 5. AGGREGATED PIPELINE EXECUTION LOOP
# =====================================================================
if __name__ == "__main__":
    # Point this to your directory where files are stored (e.g. current path)
    log_files_directory = "./Drill Logs" 
    target_files = glob.glob(os.path.join(log_files_directory, "drill_hole_*.txt"))
    
    print(f"📦 Discovered {len(target_files)} target log files for analysis.\n")
    
    all_extracted_insights = []
    
    for file_path in target_files:
        print(f"📖 Scanning: {os.path.basename(file_path)}")
        
        # Step A: Semantic Chunking (cuts along logical changes in text layout)
        semantic_segments = semantic_chunk_file(file_path)
        
        # Step B: Dynamic Window grouping (ensures text fits target token boundaries safely)
        optimized_windows = build_dynamic_sliding_windows(
            semantic_segments, 
            max_window_tokens=1000, # Conservative chunk to prevent context fragmentation
            overlap_tokens=200      # 200 token overlap cushion
        )
        
        print(f"   ├─ Extracted {len(semantic_segments)} semantic segments.")
        print(f"   ├─ Packed into {len(optimized_windows)} sliding window prompts.")
        
        # Step C: LLM Ingestion
        for idx, window in enumerate(optimized_windows):
            print(f"   │  └─ Processing window {idx+1}/{len(optimized_windows)} ({window['token_count']} tokens)...")
            extraction_result = execute_drill_log_scan(window)
            all_extracted_insights.append(extraction_result)
            
    print("\n✅ Execution Finished! Sample Data Extracted:")
    import json
    print(json.dumps(all_extracted_insights[:2], indent=2))

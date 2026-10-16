

import json
from dotenv import load_dotenv

from ChainInvokeFunctions import execute_search_mineral_conversational_output
from qdrant import get_all_text_from_qdrant

load_dotenv()

if __name__ == "__main__":
    my_data = get_all_text_from_qdrant(collection_name="optimized_windows")

    for idx, item in enumerate(my_data):
        print(f"\n[Record #{idx + 1}] Scanning {item.get('source', 'Unknown')}")
        result = execute_search_mineral_conversational_output(item)
        print(f"***result: {json.dumps(result, indent=2)}")

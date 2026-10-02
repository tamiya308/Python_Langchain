

import json
from dotenv import load_dotenv

from ChainInvokeFunctions import execute_drill_log_scan, execute_mineral_scan
from qdrant import get_all_text_from_qdrant

load_dotenv()

if __name__ == "__main__":
    my_data = get_all_text_from_qdrant(collection_name="optimized_windows")

    extraction_results = []
    for idx, item in enumerate(my_data):
        print(f"\n[Record #{idx + 1}] Scanning {item.get('source', 'Unknown')}")
        extraction_results.append(execute_mineral_scan(item))

    print(extraction_results)

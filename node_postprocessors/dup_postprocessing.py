from typing import List, Optional
from abc import abstractmethod

from llama_index import QueryBundle
from llama_index.postprocessor import BaseNodePostprocessor
from llama_index.schema import NodeWithScore


class DuplicateNodeRemoverPostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        # subtracts 1 from the score
        print("_postprocess_nodes enter now")
        unique_hashes = set()
        unique_nodes = []
        print("_postprocess_nodes process nodes")
        for node in nodes:
            node_hash = node.node.hash
            if node_hash not in unique_hashes:
                unique_hashes.add(node_hash)
                unique_nodes.append(node)
        print(len(unique_nodes))
        print("_postprocess_nodes exit")
        return unique_nodes

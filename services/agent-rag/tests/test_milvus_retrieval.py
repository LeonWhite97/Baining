import sys
from types import SimpleNamespace


class FakeMilvusClient:
    collections: dict[str, list[dict[str, object]]] = {}

    def __init__(self, uri: str) -> None:
        self.uri = uri

    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, dimension: int, metric_type: str) -> None:
        self.collections[collection_name] = []
        self.dimension = dimension
        self.metric_type = metric_type

    def upsert(self, collection_name: str, data: list[dict[str, object]]) -> None:
        existing = {record["id"]: record for record in self.collections.get(collection_name, [])}
        for record in data:
            existing[record["id"]] = record
        self.collections[collection_name] = list(existing.values())

    def search(self, collection_name: str, data: list[list[float]], limit: int, output_fields: list[str]):
        query = data[0]
        rows = []
        for record in self.collections[collection_name]:
            vector = record["vector"]
            score = sum(float(left) * float(right) for left, right in zip(query, vector, strict=False))
            entity = {field: record[field] for field in output_fields}
            rows.append({"entity": entity, "distance": score})
        rows.sort(key=lambda row: row["distance"], reverse=True)
        return [rows[:limit]]


def test_milvus_repository_seeds_and_filters_categories(monkeypatch) -> None:
    FakeMilvusClient.collections = {}
    monkeypatch.setitem(sys.modules, "pymilvus", SimpleNamespace(MilvusClient=FakeMilvusClient))

    from agent_rag.milvus_retrieval import MilvusKnowledgeRepository

    repository = MilvusKnowledgeRepository("local.db")
    repository.initialize()

    results = repository.retrieve("球高超限", ["SOP"], limit=3)

    assert results
    assert {item.category for item in results} == {"SOP"}
    assert results[0].citation

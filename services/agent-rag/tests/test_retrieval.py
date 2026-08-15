from agent_rag.retrieval import InMemoryKnowledgeRepository


def test_retrieve_filters_category_and_preserves_citation() -> None:
    repository = InMemoryKnowledgeRepository.seeded()

    results = repository.retrieve("球高超限", ["SOP"], limit=3)

    assert results
    assert results[0].category == "SOP"
    assert results[0].document_id
    assert results[0].chunk_id
    assert results[0].citation


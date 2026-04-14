"""Evolution pipeline — discover candidate new TRIZ principles from patents."""

import logging

from triz_ai.llm.client import LLMClient
from triz_ai.patents.repository import PatentRepository
from triz_ai.patents.store import CandidateParameter, CandidatePrinciple, PatentStore

logger = logging.getLogger(__name__)


def run_evolution(
    llm_client: LLMClient | None = None,
    store: PatentRepository | None = None,
    confidence_threshold: float = 0.7,
    min_cluster_size: int = 3,
) -> list[CandidatePrinciple]:
    """Run the evolution pipeline to discover candidate new principles.

    Pipeline:
    1. Filter classifications with confidence < threshold
    2. Cluster poorly-mapped patent abstracts using LLM
    3. Propose candidate principles for each cluster (min 3 patents)
    4. Store candidates in DB

    Patents are classified during ingestion, so no batch classify step is needed.

    Returns:
        List of newly created candidate principles.
    """
    if llm_client is None:
        llm_client = LLMClient()
    if store is None:
        from triz_ai.config import load_config

        config = load_config()
        store = PatentStore()
        store.init_db()
        confidence_threshold = config.evolution.review_threshold

    # Step 1: Find poorly-mapped patents (low confidence)
    all_patents = store.get_all_patents()
    poorly_mapped = []
    for patent in all_patents:
        classification = store.get_classification(patent.id)
        if classification and classification.confidence < confidence_threshold:
            poorly_mapped.append(patent)

    logger.info(
        "Found %d poorly-mapped patents (confidence < %.2f)",
        len(poorly_mapped),
        confidence_threshold,
    )

    if len(poorly_mapped) < min_cluster_size:
        logger.info("Not enough poorly-mapped patents for clustering")
        return []

    # Step 3: Cluster using LLM
    abstracts = [f"{p.title}\n{p.abstract or ''}" for p in poorly_mapped]
    try:
        clusters = llm_client.cluster_patents(abstracts)
    except Exception:
        logger.exception("Patent clustering failed")
        return []

    # Step 4: Propose candidate principles for valid clusters
    candidates = []
    next_id = store.get_next_candidate_id()

    for cluster_indices in clusters:
        if len(cluster_indices) < min_cluster_size:
            continue

        # Get patent texts for this cluster
        cluster_patents = [poorly_mapped[i] for i in cluster_indices if i < len(poorly_mapped)]
        if len(cluster_patents) < min_cluster_size:
            continue

        cluster_texts = [f"{p.title}\n{p.abstract or ''}" for p in cluster_patents]
        try:
            proposal = llm_client.propose_candidate_principle(cluster_texts)
        except Exception:
            logger.exception("Failed to propose principle for cluster")
            continue

        candidate = CandidatePrinciple(
            id=f"C{next_id}",
            name=proposal.name,
            description=proposal.description,
            evidence_patent_ids=[p.id for p in cluster_patents],
            confidence=proposal.confidence,
        )
        store.insert_candidate_principle(candidate)
        candidates.append(candidate)
        next_id += 1
        logger.info("Proposed candidate principle: %s — %s", candidate.id, candidate.name)

    logger.info("Evolution pipeline complete: %d new candidates", len(candidates))
    return candidates


def run_parameter_evolution(
    llm_client: LLMClient | None = None,
    store: PatentRepository | None = None,
    confidence_threshold: float = 0.7,
    min_cluster_size: int = 3,
) -> list[CandidateParameter]:
    """Run the parameter evolution pipeline to discover candidate new parameters.

    Pipeline:
    1. Filter classifications with low-confidence contradiction mappings
    2. Cluster poorly-mapped contradictions via LLM
    3. Propose candidate parameters for each cluster (min 3 patents)
    4. Store in candidate_parameters table

    Patents are classified during ingestion, so no batch classify step is needed.

    Returns:
        List of newly created candidate parameters.
    """
    if llm_client is None:
        llm_client = LLMClient()
    if store is None:
        from triz_ai.config import load_config

        config = load_config()
        store = PatentStore()
        store.init_db()
        confidence_threshold = config.evolution.review_threshold

    # Step 1: Find patents with low-confidence contradiction mappings
    all_patents = store.get_all_patents()
    poorly_mapped = []
    for patent in all_patents:
        classification = store.get_classification(patent.id)
        if classification and classification.confidence < confidence_threshold:
            poorly_mapped.append(patent)

    logger.info(
        "Found %d patents with low-confidence contradictions (< %.2f)",
        len(poorly_mapped),
        confidence_threshold,
    )

    if len(poorly_mapped) < min_cluster_size:
        logger.info("Not enough poorly-mapped patents for parameter clustering")
        return []

    # Step 3: Cluster using LLM
    abstracts = [f"{p.title}\n{p.abstract or ''}" for p in poorly_mapped]
    try:
        clusters = llm_client.cluster_patents(abstracts)
    except Exception:
        logger.exception("Patent clustering for parameter evolution failed")
        return []

    # Step 4: Propose candidate parameters for valid clusters
    candidates = []
    existing = store.get_pending_candidate_parameters()
    next_id = len(existing) + 1

    for cluster_indices in clusters:
        if len(cluster_indices) < min_cluster_size:
            continue

        cluster_patents = [poorly_mapped[i] for i in cluster_indices if i < len(poorly_mapped)]
        if len(cluster_patents) < min_cluster_size:
            continue

        cluster_texts = [f"{p.title}\n{p.abstract or ''}" for p in cluster_patents]
        try:
            proposal = llm_client.propose_candidate_parameter(cluster_texts)
        except Exception:
            logger.exception("Failed to propose parameter for cluster")
            continue

        candidate = CandidateParameter(
            id=f"P{next_id}",
            name=proposal.name,
            description=proposal.description,
            evidence_patent_ids=[p.id for p in cluster_patents],
            confidence=proposal.confidence,
        )
        store.insert_candidate_parameter(candidate)
        candidates.append(candidate)
        next_id += 1
        logger.info("Proposed candidate parameter: %s — %s", candidate.id, candidate.name)

    logger.info("Parameter evolution pipeline complete: %d new candidates", len(candidates))
    return candidates

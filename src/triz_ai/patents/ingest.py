"""Patent ingestion pipeline — supports txt, pdf, json formats."""

import json
import logging
import uuid
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn

from triz_ai.engine.classifier import classify
from triz_ai.llm.client import LLMClient
from triz_ai.patents.store import Patent, PatentStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".json"}


def _generate_id() -> str:
    """Generate a unique patent ID."""
    return f"LOCAL-{uuid.uuid4().hex[:12].upper()}"


def _ingest_txt(path: Path) -> list[Patent]:
    """Ingest a text file. First line is title, rest is abstract."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    lines = text.split("\n", 1)
    title = lines[0].strip()
    abstract = lines[1].strip() if len(lines) > 1 else ""
    return [
        Patent(
            id=_generate_id(),
            title=title,
            abstract=abstract,
            source="curated",
        )
    ]


def _ingest_pdf(path: Path) -> list[Patent]:
    """Ingest a PDF file using pdfplumber."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        pages_text = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

    if not pages_text:
        logger.warning("No text extracted from PDF: %s", path)
        return []

    full_text = "\n".join(pages_text)
    lines = full_text.split("\n", 1)
    title = lines[0].strip()
    abstract = lines[1].strip() if len(lines) > 1 else ""

    return [
        Patent(
            id=_generate_id(),
            title=title,
            abstract=abstract,
            source="curated",
        )
    ]


def _ingest_json(path: Path) -> list[Patent]:
    """Ingest a JSON file — expects array of patent objects."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        data = [data]

    patents = []
    for item in data:
        patent = Patent(
            id=item.get("id", _generate_id()),
            title=item["title"],
            abstract=item.get("abstract"),
            claims=item.get("claims"),
            domain=item.get("domain"),
            filing_date=item.get("filing_date"),
            source=item.get("source", "curated"),
        )
        patents.append(patent)
    return patents


def ingest_file(
    path: Path,
    store: PatentStore,
    llm_client: LLMClient | None = None,
    embed: bool = True,
    auto_classify: bool = True,
    show_progress: bool = True,
) -> tuple[list[Patent], int]:
    """Ingest a single file and store patents.

    Args:
        path: Path to the file to ingest.
        store: Patent store to save to.
        llm_client: LLM client for embeddings (created if None and embed=True).
        embed: Whether to compute and store embeddings.
        auto_classify: Whether to classify patents during ingestion.
        show_progress: Whether to show a progress bar.

    Returns:
        Tuple of (list of ingested patents, number classified).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    if ext == ".txt":
        patents = _ingest_txt(path)
    elif ext == ".pdf":
        patents = _ingest_pdf(path)
    elif ext == ".json":
        patents = _ingest_json(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if (embed or auto_classify) and llm_client is None:
        llm_client = LLMClient()

    classified_count = _store_patents(
        patents, store, llm_client, embed, auto_classify, show_progress
    )
    return patents, classified_count


def ingest_directory(
    directory: Path,
    store: PatentStore,
    llm_client: LLMClient | None = None,
    embed: bool = True,
    auto_classify: bool = True,
    show_progress: bool = True,
) -> tuple[list[Patent], int]:
    """Ingest all supported files from a directory.

    Args:
        directory: Directory to scan for patent files.
        store: Patent store to save to.
        llm_client: LLM client for embeddings.
        embed: Whether to compute and store embeddings.
        auto_classify: Whether to classify patents during ingestion.
        show_progress: Whether to show a progress bar.

    Returns:
        Tuple of (list of all ingested patents, number classified).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    if (embed or auto_classify) and llm_client is None:
        llm_client = LLMClient()

    # Collect all patents from files first
    all_patents: list[Patent] = []
    for ext in SUPPORTED_EXTENSIONS:
        for file_path in sorted(directory.glob(f"*{ext}")):
            try:
                parsed = _parse_file(file_path)
                all_patents.extend(parsed)
            except Exception:
                logger.exception("Failed to parse %s", file_path)

    # Store with progress
    classified_count = _store_patents(
        all_patents, store, llm_client, embed, auto_classify, show_progress
    )

    logger.info("Ingested %d patents from %s", len(all_patents), directory)
    return all_patents, classified_count


def _parse_file(path: Path) -> list[Patent]:
    """Parse a file into Patent objects without storing."""
    ext = path.suffix.lower()
    if ext == ".txt":
        return _ingest_txt(path)
    elif ext == ".pdf":
        return _ingest_pdf(path)
    elif ext == ".json":
        return _ingest_json(path)
    return []


def _store_patents(
    patents: list[Patent],
    store: PatentStore,
    llm_client: LLMClient | None,
    embed: bool,
    auto_classify: bool,
    show_progress: bool,
) -> int:
    """Store patents with optional embeddings, classification, and progress bar.

    Returns:
        Number of patents successfully classified.
    """
    if not patents:
        return 0

    classified_count = 0

    if show_progress and len(patents) > 1:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Ingesting patents...", total=len(patents))
            for patent in patents:
                progress.update(task, description=f"Ingesting {patent.id}")
                embedding = _get_embedding(patent, llm_client, embed)
                store.insert_patent(patent, embedding=embedding)
                if auto_classify and llm_client is not None:
                    progress.update(task, description=f"Ingesting & classifying {patent.id}")
                    if _classify_patent(patent, llm_client, store):
                        classified_count += 1
                progress.advance(task)
    else:
        for patent in patents:
            embedding = _get_embedding(patent, llm_client, embed)
            store.insert_patent(patent, embedding=embedding)
            if (
                auto_classify
                and llm_client is not None
                and _classify_patent(patent, llm_client, store)
            ):
                classified_count += 1

    return classified_count


def _get_embedding(
    patent: Patent, llm_client: LLMClient | None, embed: bool
) -> list[float] | None:
    """Get embedding for a patent, returning None on failure."""
    if not embed or llm_client is None:
        return None
    text = f"{patent.title}\n{patent.abstract or ''}"
    try:
        return llm_client.get_embedding(text)
    except Exception:
        logger.warning("Failed to embed patent %s, storing without embedding", patent.id)
        return None


def _classify_patent(patent: Patent, llm_client: LLMClient, store: PatentStore) -> bool:
    """Classify a patent during ingestion. Returns True on success."""
    text = f"{patent.title}\n{patent.abstract or ''}\n{patent.claims or ''}"
    try:
        classify(text, patent_id=patent.id, llm_client=llm_client, store=store)
        return True
    except Exception:
        logger.warning("Failed to classify patent %s, skipping classification", patent.id)
        return False

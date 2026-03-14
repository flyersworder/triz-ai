"""Patent ingestion pipeline — supports txt, pdf, json formats."""

import json
import logging
import uuid
from pathlib import Path

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
) -> list[Patent]:
    """Ingest a single file and store patents.

    Args:
        path: Path to the file to ingest.
        store: Patent store to save to.
        llm_client: LLM client for embeddings (created if None and embed=True).
        embed: Whether to compute and store embeddings.

    Returns:
        List of ingested patents.
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

    if embed and llm_client is None:
        llm_client = LLMClient()

    for patent in patents:
        embedding = None
        if embed and llm_client is not None:
            text = f"{patent.title}\n{patent.abstract or ''}"
            try:
                embedding = llm_client.get_embedding(text)
            except Exception:
                logger.warning("Failed to embed patent %s, storing without embedding", patent.id)
        store.insert_patent(patent, embedding=embedding)
        logger.info("Ingested patent: %s — %s", patent.id, patent.title)

    return patents


def ingest_directory(
    directory: Path,
    store: PatentStore,
    llm_client: LLMClient | None = None,
    embed: bool = True,
) -> list[Patent]:
    """Ingest all supported files from a directory.

    Args:
        directory: Directory to scan for patent files.
        store: Patent store to save to.
        llm_client: LLM client for embeddings.
        embed: Whether to compute and store embeddings.

    Returns:
        List of all ingested patents.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    if embed and llm_client is None:
        llm_client = LLMClient()

    all_patents = []
    for ext in SUPPORTED_EXTENSIONS:
        for file_path in sorted(directory.glob(f"*{ext}")):
            try:
                patents = ingest_file(file_path, store, llm_client=llm_client, embed=embed)
                all_patents.extend(patents)
            except Exception:
                logger.exception("Failed to ingest %s", file_path)

    logger.info("Ingested %d patents from %s", len(all_patents), directory)
    return all_patents

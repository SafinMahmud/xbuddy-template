"""Learning-resource retrieval: skill tags plus BM25, no embeddings.

JobBuddy suggests courses, docs and certifications in the Skill Gap section and in
the roadmap's learning plan. Left alone, the model invents titles and links. This
module retrieves from a small curated library (learning_resources.json) instead,
and the prompts tell the model to cite only what was retrieved.

Retrieval runs in two steps, both deterministic and free to run:
  1. Skill tags. A fixed vocabulary of skills, each with the phrases that name it
     in a job posting ("k8s", "Kubernetes"). Tags found in the query are matched
     against each resource's tags. This is the strong signal.
  2. BM25 over each resource's title, summary and tags. It only ranks the
     resources that share a tag; it never adds one on its own.

Only a shared skill tag makes a resource eligible. So a requirement like "5 years
of experience" or "bias to action" retrieves nothing rather than something
irrelevant. An earlier version let a strong BM25 match through without a tag; on
the held-out postings that returned an ML course for "quickly learning ... fast-paced"
and a Power BI course for "Power Platform", so it was removed. A wrong resource in a
roadmap is worse than none: with none, the model suggests a hands-on project.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

LIBRARY_PATH = Path(__file__).with_name("learning_resources.json")

TAG_WEIGHT = 2.0  # one shared skill tag outweighs any normalised BM25 score
BM25_K1 = 1.5
BM25_B = 0.75
_LEVEL_ORDER = {"beginner": 0, "intermediate": 1, "advanced": 2}

# Skill vocabulary: tag -> phrases that name it. Phrases are matched as whole words,
# case-insensitively. Ambiguous English words ("go", "rest", "excel", "react") are
# only matched in forms that cannot be ordinary prose.
SKILL_PHRASES: dict[str, list[str]] = {
    "python": ["python"],
    "java": ["java"],
    "go": ["golang", "go programming", "go language", r"go \(golang\)"],
    "rust": ["rust"],
    "csharp": ["c#", "csharp", ".net", "dotnet", "asp.net"],
    "javascript": ["javascript", "js", "ecmascript", "es6"],
    "typescript": ["typescript"],
    "html-css": ["html", "css", "html5", "css3", "tailwind", "responsive design"],
    "react": ["react.js", "reactjs", r"react(?!\s+(?:to|quickly|fast))"],
    "nextjs": ["next.js", "nextjs"],
    "nodejs": ["node.js", "nodejs", "node", "express.js", "expressjs"],
    "fastapi": ["fastapi"],
    "django": ["django"],
    "rest-api": ["rest api", "rest apis", "restful", "api design", "web api", "web apis",
                 "openapi", "http apis"],
    "sql": ["sql", "mysql", "t-sql", "queries", "relational database", "relational databases"],
    "postgresql": ["postgresql", "postgres"],
    "vector-database": ["vector database", "vector databases", "vector db", "vector store",
                        "pgvector", "pinecone", "weaviate", "qdrant", "chroma", "faiss",
                        "embeddings", "semantic search"],
    "kafka": ["kafka", "event streaming", "event-driven"],
    "git": ["git", "github", "gitlab", "version control"],
    "linux": ["linux", "bash", "shell scripting", "unix"],
    "docker": ["docker", "containers", "containerization", "containerisation"],
    "kubernetes": ["kubernetes", "k8s", "helm", "eks", "gke", "aks"],
    "terraform": ["terraform", "infrastructure as code", "iac", "pulumi", "cloudformation"],
    "ci-cd": ["ci/cd", "ci cd", "continuous integration", "continuous delivery",
              "continuous deployment", "github actions", "jenkins", "gitlab ci"],
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda", "sagemaker", "bedrock"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "bigquery", "vertex ai"],
    "observability": ["observability", "monitoring", "opentelemetry", "prometheus",
                      "grafana", "tracing", "logging"],
    "security": ["security", "owasp", "secure coding", "authentication", "authorization"],
    "testing": ["testing", "unit tests", "unit testing", "test automation", "pytest",
                "integration tests", "tdd", "test-driven"],
    "system-design": ["system design", "software architecture", "architecture",
                      "scalability", "scalable systems", "microservices"],
    "distributed-systems": ["distributed systems", "distributed system", "idempotency",
                            "consistency", "replication", "fault tolerance",
                            "high availability", "reconciliation"],
    "algorithms": ["algorithms", "data structures", "leetcode", "dsa", "coding interview",
                   "coding interviews", "problem solving"],
    "interviewing": ["interview", "interviews", "behavioral", "behavioural",
                     "star method", "mock interview", "mock interviews"],
    "pandas": ["pandas", "numpy", "dataframes"],
    "statistics": ["statistics", "statistical", "probability", "a/b testing", "ab testing",
                   "hypothesis testing", "experimentation"],
    "data-analysis": ["data analysis", "data analytics", "analytics", "data visualization",
                      "data visualisation", "dashboards", "reporting", "insights"],
    "excel": ["microsoft excel", "ms excel", "advanced excel", "excel skills", "spreadsheets",
              "pivot tables", "vlookup", "google sheets"],
    "power-bi": ["power bi", "powerbi", "dax"],
    "tableau": ["tableau", "looker"],
    "data-engineering": ["data engineering", "data pipelines", "data pipeline", "etl", "elt",
                         "data warehouse", "data warehousing", "snowflake", "databricks"],
    "dbt": ["dbt"],
    "spark": ["spark", "pyspark", "apache spark"],
    "airflow": ["airflow", "orchestration", "dagster", "prefect"],
    "machine-learning": ["machine learning", "ml", "scikit-learn", "sklearn", "xgboost",
                         "predictive modeling", "predictive modelling", "classification",
                         "regression models"],
    "deep-learning": ["deep learning", "neural networks", "neural network", "cnn", "cnns",
                      "transformers", "computer vision"],
    "pytorch": ["pytorch", "torch", "tensorflow", "keras"],
    "nlp": ["nlp", "natural language processing", "text classification", "spacy"],
    "llm": ["llm", "llms", "large language model", "large language models", "genai",
            "gen ai", "generative ai", "gpt", "openai", "anthropic", "claude", "gemini",
            "fine-tuning", "fine tuning", "foundation models"],
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation",
            "retrieval augmented", "retrieval-augmented", "retrieval"],
    "prompt-engineering": ["prompt engineering", "prompting", "prompt design", "prompts"],
    "agents": ["agents", "agentic", "ai agents", "tool calling", "function calling",
               "multi-agent", "autonomous agents"],
    "langgraph": ["langgraph"],
    "langchain": ["langchain", "llamaindex", "llama index"],
    "evaluation": ["evals", "llm evaluation", "model evaluation", "evaluation",
                   "llm-as-a-judge", "guardrails"],
    "mlops": ["mlops", "ml ops", "model deployment", "model serving", "mlflow",
              "experiment tracking", "model monitoring", "ml pipelines", "models to production",
              "models in production", "ml in production", "deploying models",
              "deploy models", "productionizing", "productionising"],
    "agile": ["agile", "scrum", "kanban", "sprint", "sprints"],
    "ux-design": ["ux", "ui/ux", "user experience", "user research", "figma", "wireframes",
                  "prototyping", "usability"],
    "project-management": ["project management", "project manager", "pmp", "stakeholder",
                           "stakeholders", "roadmapping", "delivery management"],
}

_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in",
    "into", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "with",
    "you", "your", "we", "our", "will", "can", "able", "must", "should", "would",
    "experience", "years", "year", "strong", "solid", "good", "great", "excellent",
    "knowledge", "understanding", "skills", "skill", "proficiency", "proficient",
    "familiarity", "familiar", "working", "work", "ability", "plus", "etc", "including",
    "such", "using", "use", "e", "g", "eg", "ie", "hands",
})


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    # A phrase is a literal unless it carries regex syntax (a lookahead or escaped
    # brackets), which a few ambiguous words need.
    body = phrase if ("(?" in phrase or "\\(" in phrase) else re.escape(phrase)
    return re.compile(rf"(?<![\w.#+-]){body}(?![\w#+]|\.\w)", re.IGNORECASE)


_TAG_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    tag: [_phrase_pattern(p) for p in phrases] for tag, phrases in SKILL_PHRASES.items()
}


class Resource(BaseModel):
    """One curated learning resource."""

    id: str
    title: str
    provider: str
    url: str
    skills: list[str] = Field(min_length=1)
    level: str
    cost: str
    format: str
    summary: str

    def as_line(self) -> str:
        return (
            f"[{self.title}]({self.url}) ({self.provider}; {self.format}; {self.level}; "
            f"{self.cost})"
        )


@dataclass(frozen=True)
class Hit:
    resource: Resource
    score: float
    matched_tags: tuple[str, ...]


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9+#.]*", (text or "").lower())
    return [w.strip(".") for w in words if w.strip(".") and w.strip(".") not in _STOPWORDS]


def tag_counts(text: str) -> Counter[str]:
    """How often each skill tag is named in a text."""
    counts: Counter[str] = Counter()
    for tag, patterns in _TAG_PATTERNS.items():
        n = sum(len(p.findall(text or "")) for p in patterns)
        if n:
            counts[tag] = n
    return counts


def detect_tags(text: str) -> set[str]:
    return set(tag_counts(text))


@lru_cache(maxsize=1)
def load_library() -> tuple[Resource, ...]:
    data = json.loads(LIBRARY_PATH.read_text("utf-8"))
    return tuple(Resource.model_validate(item) for item in data)


def library_urls() -> frozenset[str]:
    return frozenset(r.url for r in load_library())


class _BM25:
    """Plain Okapi BM25 over a fixed set of documents."""

    def __init__(self, docs: list[list[str]]):
        self.docs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avg = sum(self.lengths) / max(len(docs), 1)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        terms = set(query)
        out = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    norm = tf + BM25_K1 * (1 - BM25_B + BM25_B * length / self.avg)
                    s += self.idf[t] * tf * (BM25_K1 + 1) / norm
            out.append(s)
        return out


def _document(resource: Resource) -> list[str]:
    tags = " ".join(t.replace("-", " ") for t in resource.skills)
    return tokenize(f"{resource.title} {resource.summary} {tags} {tags}")


@lru_cache(maxsize=1)
def _index() -> _BM25:
    return _BM25([_document(r) for r in load_library()])


def retrieve(
    query: str,
    k: int = 3,
    *,
    exclude: set[str] | None = None,
    free_first: bool = True,
) -> list[Hit]:
    """Best k resources for a query (a requirement name, a skill, or posting text)."""
    library = load_library()
    tags = detect_tags(query)
    terms = tokenize(query) + [w for t in tags for w in t.split("-")]
    raw = _index().scores(terms) if terms else [0.0] * len(library)
    top = max(raw) or 1.0
    hits = []
    for resource, bm25 in zip(library, raw, strict=True):
        if exclude and resource.id in exclude:
            continue
        shared = tuple(sorted(tags & set(resource.skills)))
        if not shared:  # only a named skill makes a resource eligible
            continue
        # Prefer resources focused on the skill: share of the resource's tags matched.
        focus = len(shared) / len(resource.skills) if shared else 0.0
        score = TAG_WEIGHT * len(shared) + focus + bm25 / top
        hits.append(Hit(resource, round(score, 4), shared))

    def order(h: Hit) -> tuple:
        free = 0 if (not free_first or h.resource.cost == "free") else 1
        return (-h.score, free, _LEVEL_ORDER.get(h.resource.level, 1), h.resource.id)

    return sorted(hits, key=order)[:k]


def resources_for_skills(
    text: str, *, known: str = "", limit: int = 6
) -> list[tuple[str, Resource]]:
    """One resource per skill named in a text (a pasted posting), most-named first.

    Skills the user already has (named in `known`, such as their confirmed skills)
    go last, so the list leads with likely gaps.
    """
    counts = tag_counts(text)
    have = detect_tags(known)
    ordered = sorted(counts, key=lambda t: (t in have, -counts[t], t))
    picked: list[tuple[str, Resource]] = []
    used: set[str] = set()
    for tag in ordered:
        hits = retrieve(tag.replace("-", " "), k=1, exclude=used)
        hits = [h for h in hits if tag in h.matched_tags]
        if hits:
            picked.append((tag, hits[0].resource))
            used.add(hits[0].resource.id)
        if len(picked) >= limit:
            break
    return picked


SKILL_GAP_INSTRUCTION = (
    "LEARNING RESOURCES (retrieved from JobBuddy's curated library for the skills named "
    "in the pasted posting; skills the user already has are listed last). When you "
    "suggest a learning action, choose from this list and give its title and link. "
    "Never name a course, book or certification that is not listed, and never write any "
    "other link. If nothing listed fits a gap, suggest a small hands-on project instead."
)


def skill_gap_resources_block(posting_text: str, known: str = "", limit: int = 6) -> str:
    """Prompt block of library resources for the skills in a pasted posting, or ""."""
    picked = resources_for_skills(posting_text, known=known, limit=limit)
    if not picked:
        return ""
    lines = [f"- {tag}: {resource.as_line()}" for tag, resource in picked]
    return SKILL_GAP_INSTRUCTION + "\n" + "\n".join(lines)


def resources_for_requirements(
    names: list[str], *, per_requirement: int = 2, limit: int = 10
) -> list[tuple[str, list[Resource]]]:
    """Resources for each requirement, in the order given, without repeats."""
    out: list[tuple[str, list[Resource]]] = []
    used: set[str] = set()
    total = 0
    for name in names:
        hits = retrieve(name, k=per_requirement, exclude=used)
        if not hits:
            continue
        picked = [h.resource for h in hits][: max(limit - total, 0)]
        if not picked:
            break
        used.update(r.id for r in picked)
        total += len(picked)
        out.append((name, picked))
    return out


_URL = re.compile(r"https?://[^\s)\]>\"'<]+")


def links_in(text: str) -> list[str]:
    return [u.rstrip(".,;:") for u in _URL.findall(text or "")]


def _same_link(a: str, b: str) -> bool:
    return a.rstrip("/").lower() == b.rstrip("/").lower()


def unknown_links(text: str, allowed: set[str] | frozenset[str] | None = None) -> list[str]:
    """Links in a text that are not in the resource library (or `allowed`)."""
    known = set(library_urls()) | set(allowed or ())
    return [u for u in links_in(text) if not any(_same_link(u, k) for k in known)]


def strip_unknown_links(text: str, allowed: set[str] | None = None) -> str:
    """Remove links that are not in the library. A Markdown link keeps its text."""
    bad = set(unknown_links(text, allowed))
    if not bad:
        return text

    def md(m: re.Match[str]) -> str:
        return m.group(1) if m.group(2).rstrip(".,;:") in bad else m.group(0)

    text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", md, text)
    for url in bad:
        text = text.replace(url, "")
    return text

"""The tools the model can call, by name."""

from .job_posting import read_job_posting
from .job_search import search_jobs

TOOLS = [search_jobs, read_job_posting]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

"""Check that every link in the learning-resource library still opens.

Run it before changing the library, and now and then to catch moved pages:
    cd src && python -m agents.xbuddy.knowledge.check_links
Exits 1 if any link fails, so it can run in CI later.
"""

import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .retriever import load_library

TIMEOUT = 15
HEADERS = {"User-Agent": "Mozilla/5.0 (JobBuddy link check)"}


def status(url: str) -> int | str:
    request = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception as exc:  # noqa: BLE001 - report every failure, keep checking
        return type(exc).__name__


def main() -> int:
    library = load_library()
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda r: (r, status(r.url)), library))
    broken = [(r, s) for r, s in results if s != 200]
    for r, s in broken:
        print(f"{s}  {r.id}  {r.url}")
    print(f"{len(library) - len(broken)} of {len(library)} links OK")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())

"""Abstract base crawler and shared utilities."""
from __future__ import annotations
import abc, csv, functools, logging, time
from pathlib import Path
from typing import Any, Callable, TypeVar
from .models import CSV_COLUMNS, RentalProperty

F = TypeVar("F", bound=Callable[..., Any])

def retry_on_exception(max_attempts: int = 3, delay: float = 2.0, exceptions: tuple = (Exception,)) -> Callable[[F], F]:
    """Retry a synchronous function."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc = None
            logger = logging.getLogger(func.__module__)
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        wait = delay * attempt
                        logger.warning(f"Attempt {attempt}/{max_attempts} failed for {func.__qualname__}: {exc} — retrying in {wait}s")
                        time.sleep(wait)
            raise last_exc
        return wrapper
    return decorator

class BaseCrawler(abc.ABC):
    """Base class for all rental crawlers."""
    def __init__(self, output_csv: Path, logger: logging.Logger | None = None) -> None:
        self.output_csv = output_csv
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._seen_urls: set[str] = set()

    def load_existing_urls(self) -> set[str]:
        """Load existing URLs from CSV."""
        if not self.output_csv.exists():
            self._seen_urls = set()
            return self._seen_urls
        with self.output_csv.open("r", encoding="utf-8-sig") as fh:
            self._seen_urls = {row.get("網址", "").strip() for row in csv.DictReader(fh)}
        self.logger.info(f"Loaded {len(self._seen_urls)} existing URLs from {self.output_csv}")
        return self._seen_urls

    def is_new_url(self, url: str) -> bool:
        """Check if URL is new."""
        return url not in self._seen_urls

    def save_properties(self, properties: list[RentalProperty]) -> None:
        """Save properties to CSV."""
        if not properties: return
        file_exists = self.output_csv.exists()
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.output_csv.open("a", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()
            for prop in properties:
                writer.writerow(prop.to_csv_row())
                self._seen_urls.add(prop.url)
        self.logger.info(f"Saved {len(properties)} properties → {self.output_csv}")

    @abc.abstractmethod
    def run(self) -> list[RentalProperty]:
        """Execute the crawler."""

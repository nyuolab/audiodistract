"""Save derived scientific results with their source provenance."""
import datetime as dt
from pathlib import Path
from llm_distract.mechanism.data import write_parquet


def write_with_provenance(df, path: Path, sources: list, extra: dict) -> None:
    provenance = {
        'exported_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'exporter': 'llm_distract.mechanism.provenance',
        'sources': [{'path': str(p), 'mtime': dt.datetime.fromtimestamp(Path(p).stat().st_mtime).isoformat()}
                    for p in sources],
        **extra,
    }
    write_parquet(df, path, provenance)


"""Save rendered figures without opening the final image in read/write mode."""
import errno
from io import BytesIO
import os
from pathlib import Path
import tempfile
import time


def write_bytes_atomic(path, data, attempts=3, retry_delay=0.5):
    path = Path(path)
    if attempts < 1 or retry_delay < 0:
        raise ValueError("attempts must be positive and retry_delay nonnegative")
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        temporary = None
        try:
            # A sibling file keeps replacement on the same filesystem and leaves
            # an existing output intact if writing the new one fails.
            with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                    prefix=".rapc-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            return
        except OSError as exc:
            retryable = exc.errno in {errno.EINVAL, errno.EACCES, errno.EBUSY} or getattr(exc, "winerror", None) in {32, 33}
            if not retryable or attempt + 1 == attempts:
                raise OSError(exc.errno, f"Could not save output: {exc.strerror}", str(path)) from exc
            time.sleep(retry_delay * (attempt + 1))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def save_figure_atomic(figure, path, **kwargs):
    path = Path(path)
    image_format = kwargs.pop("format", path.suffix.lstrip("."))
    if not image_format:
        raise ValueError("A file extension or explicit format is required")
    # Pillow writes to memory; only our binary writer touches the destination.
    with BytesIO() as buffer:
        figure.savefig(buffer, format=image_format, **kwargs)
        write_bytes_atomic(path, buffer.getvalue())

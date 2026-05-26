"""I/O tools: read_file, write_file.

read_file:  Never-irreversible info-tool.
write_file: Conditional-irreversible action-tool — irreversible если файл
            существует (overwrite уничтожает старое содержимое).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from harnes.tools.registry import register_predicate
from harnes.tools.schema import BaseIrreversibility, RetryPolicy, Tool, ToolCategory

if TYPE_CHECKING:
    from harnes.tools.registry import ToolRegistry


# ============================================================
# read_file
# ============================================================


class ReadFileArgs(BaseModel):
    path: str = Field(description="Absolute or relative path to the file to read")
    max_bytes: int | None = Field(
        default=None, description="Если задано — обрезать чтение этим числом байт"
    )


class ReadFileResult(BaseModel):
    content: str
    bytes_read: int
    truncated: bool = False


def read_file_impl(args: ReadFileArgs) -> ReadFileResult:
    p = Path(args.path)
    if not p.exists():
        raise FileNotFoundError(f"{args.path} does not exist")
    if not p.is_file():
        raise IsADirectoryError(f"{args.path} is not a file")

    raw = p.read_bytes()
    truncated = False
    if args.max_bytes is not None and len(raw) > args.max_bytes:
        raw = raw[: args.max_bytes]
        truncated = True

    text = raw.decode("utf-8", errors="replace")
    return ReadFileResult(content=text, bytes_read=len(raw), truncated=truncated)


READ_FILE_TOOL = Tool(
    id="read_file",
    name="read_file",
    description="Read the contents of a file from the filesystem (UTF-8).",
    input_schema=ReadFileArgs.model_json_schema(),
    output_schema=ReadFileResult.model_json_schema(),
    base_irreversibility=BaseIrreversibility.NEVER,
    side_effects="None — reads only.",
    category=ToolCategory.INFO,
    retry_policy=RetryPolicy(),
    timeout_seconds=10.0,
    implementation_ref="harnes.tools.builtin.io.read_file_impl",
)


# ============================================================
# write_file
# ============================================================


class WriteFileArgs(BaseModel):
    path: str = Field(description="Path to write the file at")
    content: str = Field(description="UTF-8 text content")
    create_parents: bool = Field(
        default=True, description="Create parent directories if missing"
    )


class WriteFileResult(BaseModel):
    bytes_written: int
    overwritten: bool


@register_predicate("write_file_overwrites_existing")
def write_file_overwrites_existing(args: WriteFileArgs) -> bool:
    """Irreversible если запись затрёт существующий файл."""
    return Path(args.path).exists()


def write_file_impl(args: WriteFileArgs) -> WriteFileResult:
    p = Path(args.path)
    existed = p.exists()

    if args.create_parents:
        p.parent.mkdir(parents=True, exist_ok=True)

    raw = args.content.encode("utf-8")
    p.write_bytes(raw)

    return WriteFileResult(bytes_written=len(raw), overwritten=existed)


WRITE_FILE_TOOL = Tool(
    id="write_file",
    name="write_file",
    description="Write UTF-8 text to a file. Overwrites if file exists.",
    input_schema=WriteFileArgs.model_json_schema(),
    output_schema=WriteFileResult.model_json_schema(),
    base_irreversibility=BaseIrreversibility.CONDITIONAL,
    conditional_predicate="write_file_overwrites_existing",
    side_effects="Creates or overwrites a file; may create parent directories.",
    category=ToolCategory.ACTION,
    retry_policy=RetryPolicy(),
    timeout_seconds=10.0,
    implementation_ref="harnes.tools.builtin.io.write_file_impl",
)


# ============================================================
# Registration
# ============================================================


def register(registry: "ToolRegistry") -> None:
    registry.register(READ_FILE_TOOL, read_file_impl, ReadFileArgs, ReadFileResult)
    registry.register(WRITE_FILE_TOOL, write_file_impl, WriteFileArgs, WriteFileResult)

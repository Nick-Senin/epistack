"""
Демо для структурированного суммаризатора с семантическим сплиттером.
"""
import argparse
import json
import os
import sys

import dspy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module_semantic_parallel_splitter.config import configure_module_llm

from .module import summarize_text


def load_dataset_text(file_path: str, limit: int) -> str:
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if limit > 0:
            data = data[:limit]
        return "\n\n".join(item["text"] for item in data if item.get("text"))
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> int:
    parser = argparse.ArgumentParser(description="DSPy structured summarizer demo")
    parser.add_argument(
        "--chunks-file",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "datasets",
            "long_texts",
            "interview_telek_mamutov.md",
        ),
        help="Путь к входному файлу (.json с чанками или .md/.txt)",
    )
    parser.add_argument("--limit", type=int, default=6, help="Сколько чанков взять из датасета")
    parser.add_argument(
        "--parent-heading",
        default="Welcome to the DSPy Documentation.",
        help="Корневой заголовок для результата",
    )
    parser.add_argument("--max-chunk-size", type=int, default=3000)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--output-file", default="summary.md", help="Путь к файлу для сохранения результата")
    parser.add_argument("--min-chunk-len", type=int, default=100)
    parser.add_argument("--max-resplit-iters", type=int, default=3)
    args = parser.parse_args()

    configure_module_llm(use_global_config=True)

    input_text = load_dataset_text(args.chunks_file, args.limit)
    if not input_text.strip():
        print("Пустой текст для обработки.")
        return 1

    summary = summarize_text(
        input_text,
        parent_heading=args.parent_heading,
        splitter_kwargs={
            "max_chunk_size": args.max_chunk_size,
            "num_threads": args.num_threads,
            "min_chunk_len": args.min_chunk_len,
            "max_resplit_iters": args.max_resplit_iters,
        },
        num_threads=args.num_threads,
        max_depth=args.max_depth,
        output_path=args.output_file,
    )

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

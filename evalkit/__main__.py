# ai_eval/__main__.py
from pathlib import Path
from rich.console import Console
from rich.panel   import Panel
from rich.text    import Text
from rich import box
from rich.style import Style
from evaluators.image_classification import run_classification
from evaluators.object_detection    import run_detection
from evaluators.segmentation        import run_segmentation
from evaluators.speech_recognition  import run_speech_recognition
from evaluators.text_analysis       import run_text_analysis

console = Console()

def _add_default_prefix(path_str: str, default_root: str) -> str:
    """If the user forgot to type the folder prefix (models/ or data/), add it."""
    p = Path(path_str)
    return path_str if p.exists() else f"{default_root}/{path_str}"


def main():
    # ─── big “Available tasks” banner ──────────────────────────────

    kw_style   = Style(color="bright_cyan",  bold=True)   # the word: classify / detect / …
    desc_style = Style(color="bright_magenta")             # the long description

    task_text = Text()

    task_text.append("  1) ", style="bold")
    task_text.append("classify", style=kw_style)
    task_text.append("   —   ", style="bold")
    task_text.append("Image classification\n", style=desc_style)

    task_text.append("  2) ", style="bold")
    task_text.append("detect", style=kw_style)
    task_text.append("     —   ", style="bold")
    task_text.append("Object detection\n", style=desc_style)

    task_text.append("  3) ", style="bold")
    task_text.append("seg", style=kw_style)
    task_text.append("        —   ", style="bold")
    task_text.append("Semantic segmentation\n", style=desc_style)

    task_text.append("  4) ", style="bold")
    task_text.append("speech", style=kw_style)
    task_text.append("     —   ", style="bold")
    task_text.append("Speech recognition\n", style=desc_style)

    task_text.append("  5) ", style="bold")
    task_text.append("text", style=kw_style)
    task_text.append("       —   ", style="bold")
    task_text.append("Text analysis", style=desc_style)

    console.print(
        Panel(
            task_text,
            title="Available tasks",
            title_align="left",
            border_style="bright_cyan",
            padding=(1, 2),
            box=box.ROUNDED,
            expand=False
        )
    )

    tasks = {
        "1": "classify", "classify": "classify",
        "2": "detect",   "detect":   "detect",
        "3": "seg",      "seg":      "seg",
        "4": "speech",   "speech":   "speech",
        "5": "text",     "text":     "text",
    }
    choice = console.input("\nSelect task [1-5 or name]: ").strip().lower()
    task   = tasks.get(choice)
    if not task:
        console.print(f"[red]Invalid choice '{choice}'.[/red]")
        return

    if task == "classify":
        # (available model: resnet50_cifar10.pth)
        m = console.input(
            f"AI Model [bold green](available model: resnet50_cifar10.pth)[/bold green]: "
        ).strip()
        # (available data: cifar10_test)
        d = console.input(
            f"Data  folder [bold green](available data: cifar10_test)[/bold green]: "
        ).strip()
        loops = int(console.input("Loops [default 10]: ") or "10")

        m = _add_default_prefix(m, "models")
        d = _add_default_prefix(d, "data")
        run_classification(m, d, loops)

    elif task == "detect":
        # (available model: car-lp.pt)
        m = console.input(
            f"AI Model [bold green](available model: car-lp.pt)[/bold green]: "
        ).strip()
        # (available data: car_lp.yaml)
        d = console.input(
            f"Data (yaml) [bold green](available data: car_lp.yaml)[/bold green]: "
        ).strip()
        loops = int(console.input("Loops [default 10]: ") or "10")

        run_detection(m, d, loops)

    elif task == "seg":
        # (available model: road.pth)
        m = console.input(
            f"AI Model [bold green](available model: road.pth)[/bold green]: "
        ).strip()
        # (available data: road_test)
        d = console.input(
            f"Data folder [bold green](available data: road_test)[/bold green]: "
        ).strip()
        loops = int(console.input("Loops [default 10]: ") or "10")

        run_segmentation(m, d, loops)

    elif task == "speech":
        # (available model: wav2vec2-final)
        m = console.input(
            f"AI Model folder [bold green](available model: wav2vec2-final)[/bold green]: "
        ).strip()
        # (available data: librispeech_test)
        d = console.input(
            f"Data folder [bold green](available data: librispeech_test)[/bold green]: "
        ).strip()
        loops = int(console.input("Loops [default 10]: ") or "10")

        run_speech_recognition(m, d, loops)

    elif task == "text":
        # (available model: text)
        m = console.input(
            f"AI Model folder [bold green](available model: text)[/bold green]: "
        ).strip()
        # (available data: text_test)
        s = console.input(
            f"Data (csv) [bold green](available data: text_test)[/bold green]: "
        ).strip()
        loops = int(console.input("Loops [default 10]: ") or "10")

        run_text_analysis(m, s, loops)

if __name__ == "__main__":
    main()

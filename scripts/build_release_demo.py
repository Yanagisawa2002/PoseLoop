#!/usr/bin/env python3
"""Build the factual PoseLoop release video from frozen local evidence."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


WIDTH = 1280
HEIGHT = 720
BACKGROUND = "#07111f"
FOREGROUND = "#f6f7fb"
MUTED = "#aebbd0"
BLUE = "#5aa9ff"
YELLOW = "#ffd43b"
MAGENTA = "#ff4fd8"
GREEN = "#50e890"
RED = "#ff6b6b"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(
            "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"
        ),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


FONT_TITLE = _font(54, bold=True)
FONT_HEADING = _font(38, bold=True)
FONT_BODY = _font(27)
FONT_SMALL = _font(20)
FONT_METRIC = _font(45, bold=True)


def _base() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    return image, ImageDraw.Draw(image)


def _center(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((WIDTH - (box[2] - box[0])) / 2, y), text, font=font, fill=fill)


def _footer(draw: ImageDraw.ImageDraw) -> None:
    text = "XYZ-IBD RealSense development data | custom metrics | not sealed or official BOP"
    draw.rectangle((0, HEIGHT - 42, WIDTH, HEIGHT), fill="#0b1b2e")
    _center(draw, text, HEIGHT - 34, FONT_SMALL, MUTED)


def _fit(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.fit(
            source.convert("RGB"), size, method=Image.Resampling.LANCZOS
        )


def _panel(path: Path, index: int) -> Image.Image:
    with Image.open(path) as source:
        source = source.convert("RGB")
        panel_width = source.width // 3
        return source.crop(
            (index * panel_width, 20, (index + 1) * panel_width, source.height)
        )


def _image_slide(path: Path, heading: str, accent: str, detail: str) -> Image.Image:
    image, draw = _base()
    draw.text((52, 34), heading, font=FONT_HEADING, fill=accent)
    frame = _fit(path, (936, 526))
    image.paste(frame, (172, 92))
    draw.rectangle((64, 618, 1216, 664), fill="#0b1b2e")
    _center(draw, detail, 628, FONT_SMALL, FOREGROUND)
    _footer(draw)
    return image


def _title_slide() -> Image.Image:
    image, draw = _base()
    _center(draw, "PoseLoop", 172, FONT_TITLE, FOREGROUND)
    _center(draw, "Auditable RGB-D instance-to-6D pose", 248, FONT_HEADING, BLUE)
    _center(
        draw,
        "820 frozen instance predictions -> FoundationPose",
        350,
        FONT_BODY,
        MUTED,
    )
    _center(
        draw,
        "Real industrial bin-picking development evaluation",
        398,
        FONT_BODY,
        MUTED,
    )
    _footer(draw)
    return image


def _pipeline_slide() -> Image.Image:
    image, draw = _base()
    _center(draw, "One inference path", 72, FONT_HEADING, FOREGROUND)
    labels = [
        ("RGB-D", BLUE),
        ("Instance masks", YELLOW),
        ("FoundationPose", MAGENTA),
        ("6D poses", GREEN),
    ]
    x = 58
    for index, (label, color) in enumerate(labels):
        width = 242
        draw.rounded_rectangle((x, 275, x + width, 385), radius=18, fill="#10243b")
        box = draw.textbbox((0, 0), label, font=FONT_BODY)
        draw.text(
            (x + (width - box[2] + box[0]) / 2, 314),
            label,
            font=FONT_BODY,
            fill=color,
        )
        if index < len(labels) - 1:
            draw.line((x + width + 12, 330, x + width + 54, 330), fill=MUTED, width=5)
            draw.polygon(
                [
                    (x + width + 54, 330),
                    (x + width + 40, 320),
                    (x + width + 40, 340),
                ],
                fill=MUTED,
            )
        x += 306
    _center(
        draw,
        "Frozen inputs, pinned checkpoints, resumable primary execution",
        470,
        FONT_BODY,
        MUTED,
    )
    _footer(draw)
    return image


def _scene_intro(scene: int, status: str, recall: str, color: str) -> Image.Image:
    image, draw = _base()
    _center(draw, f"Scene {scene}", 190, FONT_TITLE, color)
    _center(draw, status, 282, FONT_HEADING, FOREGROUND)
    _center(draw, f"End-to-end joint recall: {recall}", 382, FONT_BODY, MUTED)
    _footer(draw)
    return image


def _triptych_slide(
    path: Path,
    scene: int,
    recall: str,
    note: str,
    color: str,
) -> Image.Image:
    image, draw = _base()
    draw.text(
        (46, 32), f"Scene {scene}: complete comparison", font=FONT_HEADING, fill=color
    )
    labels = [
        ("Instance masks", YELLOW),
        ("FoundationPose", MAGENTA),
        ("Development GT", GREEN),
    ]
    for index, (label, label_color) in enumerate(labels):
        x = 45 + index * 410
        panel = ImageOps.fit(
            _panel(path, index), (380, 214), method=Image.Resampling.LANCZOS
        )
        draw.text((x, 110), label, font=FONT_SMALL, fill=label_color)
        image.paste(panel, (x, 142))
    draw.text((64, 562), f"Joint recall {recall}", font=FONT_METRIC, fill=color)
    draw.text((500, 576), note, font=FONT_BODY, fill=FOREGROUND)
    _footer(draw)
    return image


def _summary_slide() -> Image.Image:
    image, draw = _base()
    _center(draw, "Fixed real-development result", 54, FONT_HEADING, FOREGROUND)
    metrics = [
        ("Detector F1", "0.726", YELLOW),
        ("Joint pose F1", "0.606", MAGENTA),
        ("Combined AR", "0.638", GREEN),
        ("Completion", "820 / 820", BLUE),
    ]
    for index, (label, value, color) in enumerate(metrics):
        x = 54 + (index % 2) * 614
        y = 160 + (index // 2) * 210
        draw.rounded_rectangle((x, y, x + 560, y + 164), radius=18, fill="#10243b")
        draw.text((x + 30, y + 26), label, font=FONT_BODY, fill=MUTED)
        draw.text((x + 30, y + 76), value, font=FONT_METRIC, fill=color)
    _footer(draw)
    return image


def _limitations_slide() -> Image.Image:
    image, draw = _base()
    _center(draw, "What the result does not hide", 68, FONT_HEADING, FOREGROUND)
    rows = [
        ("Development data, not a sealed or official leaderboard result", BLUE),
        ("AP75 = 0.101: precise mask boundaries remain difficult", YELLOW),
        ("Scene 25 recall = 0.437: thin crowded parts remain the main failure", RED),
        (
            "Retired geometry, SAM2, CAD, and temporal-recovery routes stay negative",
            MAGENTA,
        ),
    ]
    y = 180
    for text, color in rows:
        draw.ellipse((78, y + 8, 96, y + 26), fill=color)
        draw.text((120, y), text, font=FONT_BODY, fill=FOREGROUND)
        y += 92
    _center(
        draw,
        "Package the supported pipeline. Keep the limitations visible.",
        586,
        FONT_BODY,
        GREEN,
    )
    _footer(draw)
    return image


def _write_slides(
    directory: Path,
    raw10: Path,
    raw25: Path,
    triptych10: Path,
    triptych25: Path,
) -> list[tuple[Path, float]]:
    mask10 = directory / "mask10.jpg"
    fp10 = directory / "fp10.jpg"
    gt10 = directory / "gt10.jpg"
    mask25 = directory / "mask25.jpg"
    fp25 = directory / "fp25.jpg"
    gt25 = directory / "gt25.jpg"
    for source, index, output in [
        (triptych10, 0, mask10),
        (triptych10, 1, fp10),
        (triptych10, 2, gt10),
        (triptych25, 0, mask25),
        (triptych25, 1, fp25),
        (triptych25, 2, gt25),
    ]:
        _panel(source, index).save(output, quality=95)

    slides: list[tuple[Image.Image, float]] = [
        (_title_slide(), 5),
        (_pipeline_slide(), 5),
        (_scene_intro(10, "Strong representative scene", "0.800", GREEN), 3),
        (
            _image_slide(
                raw10, "1 / 4  Raw RGB", BLUE, "Dense same-object bin before prediction"
            ),
            4,
        ),
        (
            _image_slide(
                mask10,
                "2 / 4  Instance masks",
                YELLOW,
                "Separate detector proposals; labels remain closed",
            ),
            4,
        ),
        (
            _image_slide(
                fp10,
                "3 / 4  FoundationPose",
                MAGENTA,
                "Predicted CAD projections from frozen masks",
            ),
            4,
        ),
        (
            _image_slide(
                gt10,
                "4 / 4  Development GT",
                GREEN,
                "GT opened only after primary completion",
            ),
            4,
        ),
        (
            _triptych_slide(
                triptych10, 10, "0.800", "Strong alignment across dense gears", GREEN
            ),
            6,
        ),
        (_scene_intro(25, "Hardest clutter-sensitive scene", "0.437", RED), 3),
        (
            _image_slide(
                raw25,
                "1 / 4  Raw RGB",
                BLUE,
                "Thin overlapping parts create severe ambiguity",
            ),
            4,
        ),
        (
            _image_slide(
                mask25,
                "2 / 4  Instance masks",
                YELLOW,
                "Missed and merged boundaries remain visible",
            ),
            4,
        ),
        (
            _image_slide(
                fp25,
                "3 / 4  FoundationPose",
                MAGENTA,
                "Pose hypotheses inherit upstream ambiguity",
            ),
            4,
        ),
        (
            _image_slide(
                gt25,
                "4 / 4  Development GT",
                GREEN,
                "GT exposes missed instances and pose errors",
            ),
            4,
        ),
        (
            _triptych_slide(
                triptych25, 25, "0.437", "Failure case retained, not hidden", RED
            ),
            7,
        ),
        (_summary_slide(), 6),
        (_limitations_slide(), 6),
    ]
    outputs: list[tuple[Path, float]] = []
    for index, (slide, duration) in enumerate(slides):
        path = directory / f"slide-{index:02d}.png"
        slide.save(path, optimize=True)
        outputs.append((path, duration))
    return outputs


def _encode(slides: list[tuple[Path, float]], output: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to build the release video")
    concat = slides[0][0].parent / "concat.txt"
    lines: list[str] = []
    for path, duration in slides:
        escaped = path.resolve().as_posix().replace("'", "'\\''")
        lines.extend([f"file '{escaped}'", f"duration {duration:.3f}"])
    escaped_last = slides[-1][0].resolve().as_posix().replace("'", "'\\''")
    lines.extend(
        [
            f"file '{escaped_last}'",
            "duration 0.033",
            f"file '{escaped_last}'",
        ]
    )
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-vf",
            "fps=30,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            "-an",
            str(output),
        ],
        check=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-scene10", type=Path, required=True)
    parser.add_argument("--raw-scene25", type=Path, required=True)
    parser.add_argument("--triptych-scene10", type=Path, required=True)
    parser.add_argument("--triptych-scene25", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poster", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    for path in [
        args.raw_scene10,
        args.raw_scene25,
        args.triptych_scene10,
        args.triptych_scene25,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    with tempfile.TemporaryDirectory(prefix="poseloop-demo-") as temporary:
        directory = Path(temporary)
        slides = _write_slides(
            directory,
            args.raw_scene10,
            args.raw_scene25,
            args.triptych_scene10,
            args.triptych_scene25,
        )
        args.poster.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(slides[7][0]) as poster:
            poster.convert("RGB").save(args.poster, quality=92, optimize=True)
        _encode(slides, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

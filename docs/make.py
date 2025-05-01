#!/usr/bin/env python3
import os
import subprocess
import sys
import shutil
from pathlib import Path


def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent.absolute()
    docs_dir = project_root / "docs"
    src_dir = project_root / "src" / "ttmerge"

    # Clean up existing docs directory if it exists
    if docs_dir.exists():
        for item in docs_dir.iterdir():
            # Keep .git and other hidden files/directories
            if not item.name.startswith("."):
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    else:
        docs_dir.mkdir(exist_ok=True)

    # Configure pdoc and generate documentation directly in docs directory
    # with updated syntax for newer pdoc versions
    subprocess.run(
        [
            "pdoc",
            "--output-directory",
            str(docs_dir),
            "ttmerge",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(project_root / "src")},
    )

    # Create .nojekyll file to prevent GitHub Pages from using Jekyll
    (docs_dir / ".nojekyll").touch()

    # Generate sitemap.xml
    with (docs_dir / "sitemap.xml").open("w", newline="\n") as f:
        f.write(
            """<?xml version="1.0" encoding="utf-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.sitemaps.org/schemas/sitemap/0.9 http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd">"""
        )

        base_url = "https://rbertolissi.github.io/ttmerge/"

        # Add URLs for all HTML files
        for file in docs_dir.glob("**/*.html"):
            if file.name.startswith("_"):
                continue

            # Get relative path and convert to URL
            relative_path = file.relative_to(docs_dir).as_posix()
            if file.name == "index.html":
                relative_path = str(file.relative_to(docs_dir).parent).replace(".", "")
                if relative_path:
                    relative_path += "/"
            else:
                relative_path = relative_path

            f.write(f"\n<url><loc>{base_url}{relative_path}</loc></url>")

        f.write("\n</urlset>")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
LaTeX to PDF renderer using Python.
Requires: pdflatex (from TeXLive or MiKTeX)
"""

import sys
import os
import tempfile
import subprocess
import json
from pathlib import Path


def render_latex_to_pdf(latex_content: str, output_path: str) -> dict:
    """
    Render LaTeX content to PDF.
    
    Args:
        latex_content: LaTeX source code
        output_path: Path where PDF should be saved
        
    Returns:
        dict with 'success' (bool) and 'error' (str if failed)
    """
    # Create temporary directory for LaTeX compilation
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tex_file = tmpdir_path / 'document.tex'
        
        # Write LaTeX content to temporary file
        tex_file.write_text(latex_content, encoding='utf-8')

        # Prefer xelatex (better Unicode/中文支持); fallback to pdflatex
        def find_latex_cmd() -> tuple[str, str]:
            # (command, name_for_error)
            # 1) Try xelatex
            xelatex_cmd = 'xelatex'
            if sys.platform == 'win32':
                xelatex_paths = [
                    r'C:\texlive\2024\bin\win32\xelatex.exe',
                    r'C:\texlive\2023\bin\win32\xelatex.exe',
                    r'C:\Program Files\MiKTeX\miktex\bin\x64\xelatex.exe',
                    r'C:\Program Files (x86)\MiKTeX\miktex\bin\xelatex.exe',
                ]
                for p in xelatex_paths:
                    if os.path.exists(p):
                        xelatex_cmd = p
                        break

            try:
                # Quick availability check
                subprocess.run([xelatex_cmd, '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return xelatex_cmd, 'xelatex'
            except Exception:
                # 2) Fallback to pdflatex
                pdflatex_cmd = 'pdflatex'
                if sys.platform == 'win32':
                    pdflatex_paths = [
                        r'C:\texlive\2024\bin\win32\pdflatex.exe',
                        r'C:\texlive\2023\bin\win32\pdflatex.exe',
                        r'C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe',
                        r'C:\Program Files (x86)\MiKTeX\miktex\bin\pdflatex.exe',
                    ]
                    for p in pdflatex_paths:
                        if os.path.exists(p):
                            pdflatex_cmd = p
                            break
                return pdflatex_cmd, 'pdflatex'

        latex_cmd, latex_name = find_latex_cmd()

        try:
            # Run LaTeX twice (first pass for references, second for final output)
            for _ in range(2):
                result = subprocess.run(
                    [latex_cmd, '-interaction=nonstopmode', '-output-directory', str(tmpdir_path), str(tex_file)],
                    capture_output=True,
                    text=True,
                    cwd=tmpdir_path,
                    timeout=60
                )
                if result.returncode != 0:
                    return {
                        'success': False,
                        'error': f'{latex_name} failed:\n{result.stderr}\n{result.stdout}'
                    }
            
            # Check if PDF was generated
            pdf_file = tmpdir_path / 'document.pdf'
            if not pdf_file.exists():
                return {
                    'success': False,
                    'error': 'PDF file was not generated'
                }
            
            # Copy PDF to output path
            import shutil
            shutil.copy2(pdf_file, output_path)
            
            return {'success': True}
            
        except FileNotFoundError:
            return {
                'success': False,
                'error': f'LaTeX engine not found. Please install TeXLive or MiKTeX (xelatex / pdflatex).'
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'{latex_name} timed out.'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            }


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 3:
        print(json.dumps({
            'success': False,
            'error': 'Usage: render_latex.py <latex_content_file> <output_pdf_path>'
        }))
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_path = sys.argv[2]
    
    # Read LaTeX content
    try:
        latex_content = Path(input_file).read_text(encoding='utf-8')
    except Exception as e:
        print(json.dumps({
            'success': False,
            'error': f'Failed to read input file: {str(e)}'
        }))
        sys.exit(1)
    
    # Render to PDF
    result = render_latex_to_pdf(latex_content, output_path)
    print(json.dumps(result))
    
    sys.exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()


"""Create a desktop shortcut for PaperCore GUI.

Run once: python create_shortcut.py
"""

from pathlib import Path
import sys
import os


def create_shortcut():
    try:
        import winshell
    except ImportError:
        # Fallback: use VBScript to create shortcut (no extra deps)
        _create_via_vbs()
        return

    desktop = winshell.desktop()
    link_path = os.path.join(desktop, "PaperCore.lnk")

    winshell.CreateShortcut(
        Path=link_path,
        Target=sys.executable.replace("python.exe", "pythonw.exe"),
        Arguments=f'"{Path(__file__).parent / "papercore_gui.py"}"',
        StartIn=str(Path(__file__).parent),
        Description="PaperCore - Academic PDF Converter",
    )
    print(f"Shortcut created: {link_path}")


def _create_via_vbs():
    """Create shortcut using built-in Windows VBScript (no extra pip packages)."""
    script_dir = Path(__file__).parent.resolve()
    gui_path = script_dir / "papercore_gui.py"

    # Find pythonw.exe
    pythonw = Path(sys.executable).parent / "pythonw.exe"
    if not pythonw.exists():
        pythonw = Path(sys.executable)  # fallback to python.exe

    desktop = Path.home() / "Desktop"
    link_path = desktop / "PaperCore.lnk"

    vbs = f'''Set ws = CreateObject("WScript.Shell")
Set sc = ws.CreateShortcut("{link_path}")
sc.TargetPath = "{pythonw}"
sc.Arguments = """{gui_path}"""
sc.WorkingDirectory = "{script_dir}"
sc.Description = "PaperCore - Academic PDF Converter"
sc.Save
'''

    vbs_path = script_dir / "_make_shortcut.vbs"
    vbs_path.write_text(vbs, encoding="utf-8")

    os.system(f'cscript //nologo "{vbs_path}"')
    vbs_path.unlink(missing_ok=True)

    if link_path.exists():
        print(f"Shortcut created: {link_path}")
    else:
        print("Failed to create shortcut. You can manually create one:")
        print(f"  Target:  {pythonw}")
        print(f'  Args:    "{gui_path}"')
        print(f"  Start in: {script_dir}")


if __name__ == "__main__":
    create_shortcut()

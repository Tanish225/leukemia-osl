"""Download the shared Google Drive dataset folder into ./data  (needs: pip install gdown).
If it fails (Drive quota / 50-file folder limit), download the folder manually in the browser instead - see README."""
import subprocess, sys
URL = "https://drive.google.com/drive/folders/1mzafQpQyEorr0YnryD7xwwIDhFS-vKkW"
sys.exit(subprocess.call([sys.executable, "-m", "gdown", "--folder", URL, "-O", "data", "--remaining-ok"]))

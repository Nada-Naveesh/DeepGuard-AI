

import sys
from pathlib import Path

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from app import main

if __name__ == "__main__":
    print("=" * 80)
    print("DeepGuard AI: Explainable Deepfake Image Detection System")
    print("=" * 80)
    main()

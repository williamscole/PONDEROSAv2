import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ponderosa.config import PonderosaConfig

config1 = PonderosaConfig.from_dict({"files": {"fam": "data/test1.fam",
                                        "ages": "data/age1.txt",
                                        "ibd": "data/segments1.txt",
                                        "ibd_caller": "phasedibd"},
                                        })


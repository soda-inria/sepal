from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from pykeen.triples import TriplesFactory


class DataLoader:
    def __init__(self, triples_dir: Path) -> None:
        self.triples_dir = triples_dir
        self.load_metadata()
        return

    def load_metadata(self) -> None:
        with open(f"{self.triples_dir}/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        for key, value in metadata.items():
            setattr(self, key, value)
        return

    def get_triples_factory(self, create_inverse_triples=True) -> TriplesFactory:
        triples = np.load(self.triples_dir / "triplets.npy")
        df = pd.DataFrame(triples).astype(np.int32)
        tf = TriplesFactory(
            df.to_numpy(),
            self.entity_to_idx,
            self.rel_to_idx,
            create_inverse_triples=create_inverse_triples,
        )
        return tf
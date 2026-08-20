import importlib.util
from pathlib import Path


PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

RECOMMENDER_FILE = (
    PROJECT_DIR
    / "SharePoint-Recommender"
    / "recommender.py"
)


spec = importlib.util.spec_from_file_location(
    "recommender",
    RECOMMENDER_FILE
)

recommender = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    recommender
)


def test_load_data():

    (
        interactions,
        resources,
        neighbors,
        popularity
    ) = recommender.load_data()

    assert len(interactions) > 0
    assert len(resources) > 0
    assert len(neighbors) > 0
    assert len(popularity) > 0
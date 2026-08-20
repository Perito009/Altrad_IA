"""
Tests simples du projet Altrad_IA.

L'objectif est de vérifier que les données générées
respectent la structure attendue par le pipeline.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CHEMINS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "processed"


INTERACTIONS_FILE = (
    PROCESSED_DIR
    / "interactions.parquet"
)

RESOURCES_FILE = (
    PROCESSED_DIR
    / "resources.parquet"
)

NEIGHBORS_FILE = (
    PROCESSED_DIR
    / "user_neighbors.parquet"
)

POPULARITY_FILE = (
    PROCESSED_DIR
    / "resource_popularity.parquet"
)


# ============================================================
# TEST INTERACTIONS
# ============================================================

def test_interactions_file_exists():

    assert INTERACTIONS_FILE.exists()


def test_interactions_columns():

    df = pd.read_parquet(
        INTERACTIONS_FILE
    )

    required_columns = {
        "user",
        "resource"
    }

    assert required_columns.issubset(
        df.columns
    )


def test_interactions_not_empty():

    df = pd.read_parquet(
        INTERACTIONS_FILE
    )

    assert len(df) > 0


# ============================================================
# TEST RESOURCES
# ============================================================

def test_resources_file_exists():

    assert RESOURCES_FILE.exists()


def test_resources_columns():

    df = pd.read_parquet(
        RESOURCES_FILE
    )

    required_columns = {
        "resource",
        "site",
        "sous-site",
        "bibliothèque",
        "liste"
    }

    assert required_columns.issubset(
        df.columns
    )


# ============================================================
# TEST SIMILARITY
# ============================================================

def test_neighbors_file_exists():

    assert NEIGHBORS_FILE.exists()


def test_neighbors_columns():

    df = pd.read_parquet(
        NEIGHBORS_FILE
    )

    required_columns = {
        "user",
        "similar_user",
        "similarity"
    }

    assert required_columns.issubset(
        df.columns
    )


def test_similarity_range():

    df = pd.read_parquet(
        NEIGHBORS_FILE
    )

    assert (
        df["similarity"] >= 0
    ).all()

    assert (
        df["similarity"] <= 1
    ).all()


# ============================================================
# TEST POPULARITY
# ============================================================

def test_popularity_file_exists():

    assert POPULARITY_FILE.exists()


def test_popularity_columns():

    df = pd.read_parquet(
        POPULARITY_FILE
    )

    required_columns = {
        "resource",
        "support_users",
        "popularity_score"
    }

    assert required_columns.issubset(
        df.columns
    )


def test_popularity_range():

    df = pd.read_parquet(
        POPULARITY_FILE
    )

    assert (
        df["popularity_score"] >= 0
    ).all()

    assert (
        df["popularity_score"] <= 1
    ).all()
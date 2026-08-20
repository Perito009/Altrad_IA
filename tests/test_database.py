import sqlite3
from pathlib import Path


PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


def test_sqlite_available():

    connection = sqlite3.connect(
        ":memory:"
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE test (
            id INTEGER PRIMARY KEY,
            value TEXT
        )
        """
    )

    cursor.execute(
        """
        INSERT INTO test(value)
        VALUES (?)
        """,
        ("Altrad_IA",)
    )

    result = cursor.execute(
        """
        SELECT value
        FROM test
        """
    ).fetchone()

    connection.close()

    assert result[0] == "Altrad_IA"
from alembic.config import Config
from alembic.script import ScriptDirectory


def test_migration_revision_ids_fit_default_alembic_version_column() -> None:
    scripts = ScriptDirectory.from_config(Config("alembic.ini"))

    revisions = list(scripts.walk_revisions(base="base", head="heads"))

    assert revisions
    assert all(len(revision.revision) <= 32 for revision in revisions)
    assert len(scripts.get_heads()) == 1

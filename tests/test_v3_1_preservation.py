import subprocess

import pytest

from v3_1_preservation_check import examine


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path, text=True)
    git('init', '-q')
    git('config', 'user.name', 'Preservation test')
    git('config', 'user.email', 'test@example.invalid')
    (tmp_path / 'README.md').write_text('baseline overview\n')
    (tmp_path / 'old_model.py').write_text('frozen = True\n')
    git('add', '.')
    git('commit', '-qm', 'Frozen baseline')
    return tmp_path, git, git('rev-parse', 'HEAD').strip()


def test_allows_new_version_and_project_memory(repo):
    root, git, baseline = repo
    (root / 'README.md').write_text('current overview\n')
    (root / 'scripts').mkdir()
    (root / 'scripts/v3_1_new.py').write_text('new = True\n')
    git('add', '.')
    assert examine(root, baseline)['passed']


def test_rejects_staged_change_even_when_worktree_restored(repo):
    root, git, baseline = repo
    original = (root / 'old_model.py').read_text()
    (root / 'old_model.py').write_text('frozen = False\n')
    git('add', 'old_model.py')
    (root / 'old_model.py').write_text(original)
    result = examine(root, baseline)
    assert not result['passed']
    assert result['views']['index']['protected_changes'] == ['old_model.py']
    assert result['views']['worktree']['protected_changes'] == []


def test_rejects_unstaged_removal(repo):
    root, _, baseline = repo
    (root / 'old_model.py').unlink()
    result = examine(root, baseline)
    assert not result['passed']
    assert result['views']['worktree']['protected_changes'] == ['old_model.py']


def test_rejects_new_code_in_old_namespace(repo):
    root, git, baseline = repo
    (root / 'src/photdis/v3').mkdir(parents=True)
    (root / 'src/photdis/v3/new.py').write_text('changes_old_model = True\n')
    git('add', '.')
    result = examine(root, baseline)
    assert not result['passed']
    assert result['views']['index']['protected_changes'] == ['src/photdis/v3/new.py']

#!/usr/bin/env python3
"""
datool - Analyze student project participation using git and GitHub.

This tool analyzes git repositories to attribute code contributions to students.
It uses git blame with copy detection (-C -C) to track code movement, and parses
commit messages for Co-authored-by trailers to detect collaborative work.

Key features:
- Distinguishes solo work (Alone) from collaborative work (Collab)
- Filters files using include/exclude glob patterns
- Only counts non-whitespace changes (lines, commits, PRs)
- Supports GitHub PR statistics via gh CLI
- Caches results for faster repeated runs

Usage:
    datool /path/to/repo              # Local git analysis only
    datool --github /path/to/repo     # Include GitHub PR stats
    datool --init /path/to/repo       # Initialize repo with workflow files
    datool --clear-cache .            # Clear cached data
"""

import argparse
import atexit
import json
import os
import pickle
import platform
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterator, cast

# Global cache dictionary: cache_key -> stdout string
_cache: dict[str, str] = {}
# Global PR cache: (repo_path, pr_number) -> PullRequest for closed PRs
_pr_cache: dict[tuple[str, int], "PullRequest"] = {}
_cache_dirty: bool = False


class CommitNotFoundError(Exception):
    """Raised when a commit cannot be found in the repository."""

    def __init__(self, commit: str, repo_path: Path) -> None:
        self.commit = commit
        self.repo_path = repo_path
        super().__init__(f"Commit not found: {commit} in {repo_path}")


class FileNotFoundInRepoError(Exception):
    """Raised when a file cannot be found in the repository at a given commit."""

    def __init__(self, file_path: str, commit: str, repo_path: Path) -> None:
        self.file_path = file_path
        self.commit = commit
        self.repo_path = repo_path
        super().__init__(
            f"File not found: {file_path} at commit {commit[:8]} in {repo_path}"
        )


class ExecutableNotFoundError(Exception):
    """Raised when an executable command cannot be found."""

    def __init__(self, command: str) -> None:
        self.command = command
        if command == "gh":
            msg = (
                "GitHub CLI (gh) not found. Install it from https://cli.github.com/ "
                "and run 'gh auth login' to authenticate."
            )
        else:
            msg = f"Executable not found: {command}"
        super().__init__(msg)


class StudentsConfigError(Exception):
    """Raised when the students config file is invalid or cannot be found."""

    pass


class GitHubApiError(Exception):
    """Raised when a GitHub API call fails."""

    pass


class UnknownAuthorError(Exception):
    """Raised when an author is not found in the student config."""

    def __init__(
        self,
        name: str,
        email: str,
        context_type: str,
        context_detail: str,
    ) -> None:
        self.name = name
        self.email = email
        self.context_type = context_type
        self.context_detail = context_detail
        super().__init__(
            f"Unknown author '{name} <{email}>' not in student config. "
            f"Found in {context_type}: {context_detail}"
        )


@dataclass(eq=False)
class Author:
    """Represents a Git author."""

    name: str
    email: str
    github_username: str
    student_id: str = ""
    other_names: list[str] | None = None
    other_emails: list[str] | None = None
    allow_auto_update: bool = True

    # Registry of Author instances, keyed by (name, email)
    _registry: ClassVar[dict[tuple[str, str], "Author"]] = {}

    def __post_init__(self) -> None:
        if self.other_names is None:
            self.other_names = []
        if self.other_emails is None:
            self.other_emails = []

    def __str__(self) -> str:
        return f"{self.name} <{self.email}>"

    def __hash__(self) -> int:
        return hash((self.name, self.email))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Author):
            return NotImplemented
        return self.name == other.name and self.email == other.email

    @staticmethod
    def get(
        name: str,
        email: str,
        context_type: str,
        context_detail: str,
    ) -> "Author":
        """
        Get a shared Author instance by name and email.

        The Author must already be registered (from the student config).
        If not found, raises UnknownAuthorError with context about where
        the unknown author was encountered.

        Args:
            name: The author's name
            email: The author's email
            context_type: Type of context where author was found (e.g., 'commit', 'file')
            context_detail: Detail about the context (e.g., commit hash, file path)

        Returns:
            The shared Author instance

        Raises:
            UnknownAuthorError: If the author is not in the registry
        """
        key = (name, email)
        if key not in Author._registry:
            raise UnknownAuthorError(
                name=name,
                email=email,
                context_type=context_type,
                context_detail=context_detail,
            )
        return Author._registry[key]

    @staticmethod
    def clear_registry() -> None:
        """Clear the Author registry."""
        Author._registry.clear()

    @staticmethod
    def register(author: "Author") -> "Author":
        """
        Register an Author instance in the registry.

        If an Author with the same (name, email) already exists, returns the existing one.
        Otherwise registers and returns the provided Author.

        Args:
            author: The Author instance to register

        Returns:
            The registered Author instance (may be different from input if already existed)
        """
        key = (author.name, author.email)
        if key not in Author._registry:
            Author._registry[key] = author
        return Author._registry[key]

    @staticmethod
    def register_with_aliases(author: "Author") -> "Author":
        """
        Register an Author and all their alternate identities.

        Creates registry entries for all combinations of:
        - Primary name + primary email
        - Primary name + each other_email
        - Each other_name + primary email
        - Each other_name + each other_email

        All aliases point to the same canonical Author instance.

        Args:
            author: The Author instance to register

        Returns:
            The registered Author instance
        """
        # Register the primary identity
        registered = Author.register(author)

        # All names and emails to register
        all_names = [author.name] + (author.other_names or [])
        all_emails = [author.email] + (author.other_emails or [])

        # Register all combinations
        for name in all_names:
            for email in all_emails:
                key = (name, email)
                if key not in Author._registry:
                    Author._registry[key] = registered

        return registered


@dataclass
class PullRequestReview:
    """Represents a review on a pull request."""

    reviewer_username: str
    state: str  # APPROVED, CHANGES_REQUESTED, COMMENTED, DISMISSED, PENDING
    submitted_at: str


@dataclass
class PullRequestCommit:
    """Represents a commit in a pull request."""

    hash: str
    author_username: str
    author_email: str
    message: str
    date: str
    co_authors: list[str]  # List of co-author usernames


@dataclass
class PullRequestFile:
    """Represents a file changed in a pull request."""

    path: str
    additions: int
    deletions: int
    has_non_whitespace_changes: bool = True  # Default True, set by diff parsing


@dataclass
class PullRequest:
    """Represents a GitHub pull request."""

    number: int
    author_username: str
    created_at: str
    state: str  # OPEN, CLOSED, or MERGED
    merged_at: str | None
    merged_by_username: str | None
    commits: list[PullRequestCommit]
    reviews: list[PullRequestReview]
    files: list[PullRequestFile]  # Files changed in this PR with line counts

    @property
    def is_merged(self) -> bool:
        """Check if the PR was merged."""
        return self.state == "MERGED"

    @property
    def is_open(self) -> bool:
        """Check if the PR is still open."""
        return self.state == "OPEN"

    def get_approvers(self) -> list[str]:
        """Get list of usernames who approved this PR."""
        return [r.reviewer_username for r in self.reviews if r.state == "APPROVED"]

    def get_change_requesters(self) -> list[str]:
        """Get list of usernames who requested changes on this PR."""
        return [
            r.reviewer_username for r in self.reviews if r.state == "CHANGES_REQUESTED"
        ]

    def get_commenters(self) -> list[str]:
        """Get list of usernames who commented on this PR."""
        return [r.reviewer_username for r in self.reviews if r.state == "COMMENTED"]

    def get_commit_authors(self) -> set[str]:
        """Get set of usernames who authored commits in this PR."""
        return {c.author_username for c in self.commits if c.author_username}


@dataclass
class GitHubStats:
    """GitHub PR statistics for a student."""

    # PR.Alone: PRs where author did all commits without co-authors
    prs_alone_merged: int = 0
    prs_alone_open: int = 0
    # PR.Collab: PRs with commits that have co-authors
    prs_collab_merged: int = 0
    prs_collab_open: int = 0
    # Review stats
    approvals_given: int = 0
    change_requests_given: int = 0


@dataclass
class StudentsConfig:
    """Represents the students configuration file."""

    students: list[Author]
    ignore: list[Author]
    include_patterns: list[str]
    exclude_patterns: list[str]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "StudentsConfig":  # noqa: C901
        """
        Create a StudentsConfig from a dict (parsed JSON).

        Args:
            data: The parsed JSON dict from the config file

        Returns:
            StudentsConfig instance

        Raises:
            StudentsConfigError: If the config is invalid
        """
        # Validate students list
        if "students" not in data:
            raise StudentsConfigError("Config missing 'students' field")
        students_raw = data["students"]
        if not isinstance(students_raw, list):
            raise StudentsConfigError("'students' must be a list")

        students: list[Author] = []
        for i, student in enumerate(students_raw):  # type: ignore[arg-type]
            if not isinstance(student, dict):
                raise StudentsConfigError(f"Student entry {i} must be an object")
            s: dict[str, Any] = student  # type: ignore[assignment]
            if "id" not in s or not s["id"]:
                raise StudentsConfigError(f"Student entry {i} missing 'id'")
            if "name" not in s or not s["name"]:
                raise StudentsConfigError(f"Student entry {i} missing 'name'")
            if "email" not in s or not s["email"]:
                raise StudentsConfigError(f"Student entry {i} missing 'email'")
            if "github_username" not in s:
                raise StudentsConfigError(
                    f"Student entry {i} missing 'github_username'"
                )

            students.append(
                Author.register_with_aliases(
                    Author(
                        name=str(s["name"]),
                        email=str(s["email"]),
                        github_username=str(s["github_username"]),
                        student_id=str(s["id"]),
                        other_names=list(s.get("other_names") or []),
                        other_emails=list(s.get("other_emails") or []),
                        allow_auto_update=bool(s.get("allow_auto_update", True)),
                    )
                )
            )

        # Validate ignore list
        if "ignore" not in data:
            raise StudentsConfigError("Config missing 'ignore' field")
        ignore_raw = data["ignore"]
        if not isinstance(ignore_raw, list):
            raise StudentsConfigError("'ignore' must be a list")

        ignore: list[Author] = []
        for i, entry in enumerate(ignore_raw):  # type: ignore[arg-type]
            if not isinstance(entry, dict):
                raise StudentsConfigError(f"Ignore entry {i} must be an object")
            e: dict[str, Any] = entry  # type: ignore[assignment]
            if "name" not in e or not e["name"]:
                raise StudentsConfigError(f"Ignore entry {i} missing 'name'")
            # email can be empty for ignore entries, but must exist
            if "email" not in e:
                raise StudentsConfigError(f"Ignore entry {i} missing 'email'")

            ignore.append(
                Author.register_with_aliases(
                    Author(
                        name=str(e["name"]),
                        email=str(e.get("email") or ""),
                        github_username=str(e.get("github_username") or ""),
                        other_names=list(e.get("other_names") or []),
                        other_emails=list(e.get("other_emails") or []),
                        allow_auto_update=bool(e.get("allow_auto_update", True)),
                    )
                )
            )

        # Parse files include/exclude patterns
        files_config = data.get("files", {})
        if not isinstance(files_config, dict):
            raise StudentsConfigError("'files' must be an object")

        include_patterns: list[str] = []
        exclude_patterns: list[str] = []

        if "include" in files_config:
            if not isinstance(files_config["include"], list):
                raise StudentsConfigError("'files.include' must be a list")
            include_patterns = [
                str(p) for p in cast(list[Any], files_config["include"])
            ]

        if "exclude" in files_config:
            if not isinstance(files_config["exclude"], list):
                raise StudentsConfigError("'files.exclude' must be a list")
            exclude_patterns = [
                str(p) for p in cast(list[Any], files_config["exclude"])
            ]

        return StudentsConfig(
            students=students,
            ignore=ignore,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )


@dataclass
class Commit:
    """Represents a git commit."""

    hash: str
    author: Author
    date: str
    message_lines: list[str]

    @staticmethod
    def get(commit_hash: str, repo: "Repo") -> "Commit":
        """
        Get a Commit instance for the given hash.

        Args:
            commit_hash: The commit hash (full or short)
            repo: The Repo instance

        Returns:
            Commit instance with hash and message lines

        Raises:
            CommitNotFoundError: If the commit does not exist
        """
        # Resolve to full hash
        try:
            resolved_hash = git(
                "-C",
                str(repo.path),
                "rev-parse",
                commit_hash,
                cache_key=f"rev-parse:{repo.repo_id}:{commit_hash}",
            ).strip()
        except subprocess.CalledProcessError:
            raise CommitNotFoundError(commit_hash, repo.path)

        # Get commit info with author details
        # Format: author_name%n author_email%n date%n message_body
        stdout = git(
            "-C",
            str(repo.path),
            "show",
            "-s",
            "--format=%aN%n%aE%n%as%n%B",
            resolved_hash,
            cache_key=f"commit:{resolved_hash}",
        )

        lines = stdout.splitlines()
        author_name = lines[0] if len(lines) > 0 else ""
        author_email = lines[1] if len(lines) > 1 else ""
        commit_date = lines[2] if len(lines) > 2 else ""
        author = Author.get(
            name=author_name,
            email=author_email,
            context_type="commit",
            context_detail=resolved_hash[:8],
        )

        # Message starts at line 3
        message_lines = lines[3:] if len(lines) > 3 else []
        while message_lines and not message_lines[-1]:
            message_lines.pop()

        return Commit(
            hash=resolved_hash,
            author=author,
            date=commit_date,
            message_lines=message_lines,
        )

    def get_co_authors(self) -> list[Author]:
        """
        Parse Co-authored-by trailers from the commit message.

        Follows GitHub's format for commits with multiple authors:
        https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/creating-a-commit-with-multiple-authors

        The format is: Co-authored-by: name <email>

        This method is forgiving and tries multiple patterns:
        1. Precise: Co-authored-by: Name <email@example.com>
        2. Without angle brackets: Co-authored-by: Name email@example.com
        3. Loose: co-authored[-by] Name email@example.com (no colon required)

        Returns:
            List of Author instances parsed from Co-authored-by lines, or empty list
        """
        import re

        co_authors: list[Author] = []

        # Pattern 1: Precise - Co-authored-by: Name <email@example.com>
        precise_pattern = re.compile(
            r"^Co-authored-by:\s*(.+?)\s*<([^>]+)>\s*$", re.IGNORECASE
        )

        # Pattern 2: Without brackets - Co-authored-by: Name email@example.com
        no_brackets_pattern = re.compile(
            r"^Co-authored-by:\s*(.+?)\s+(\S+@\S+)\s*$", re.IGNORECASE
        )

        # Pattern 3: Loose - co-authored[-by] Name email (word starts with "co-authored")
        loose_pattern = re.compile(
            r"^co-authored(?:-by)?[:\s]+(.+?)\s+(\S+@\S+)\s*$", re.IGNORECASE
        )

        for line in self.message_lines:
            stripped = line.strip()
            name: str | None = None
            email: str | None = None

            # Try patterns in order of precision
            match = precise_pattern.match(stripped)
            if match:
                name = match.group(1).strip()
                email = match.group(2).strip()
            else:
                match = no_brackets_pattern.match(stripped)
                if match:
                    name = match.group(1).strip()
                    email = match.group(2).strip()
                else:
                    match = loose_pattern.match(stripped)
                    if match:
                        name = match.group(1).strip()
                        email = match.group(2).strip()

            if name and email:
                # Clean up email if it has stray angle brackets
                email = email.strip("<>")
                co_authors.append(
                    Author.get(
                        name=name,
                        email=email,
                        context_type="co-author in commit",
                        context_detail=self.hash[:8],
                    )
                )

        return co_authors


@dataclass
class TrackedFile:
    """Represents a versioned file in the Git repository with blame information."""

    path: str
    lines: list[tuple[Commit, str]]

    @staticmethod
    def files(
        pattern: str, repo: "Repo", commit: str = "HEAD"
    ) -> Iterator["TrackedFile"]:
        """
        Yield TrackedFile instances matching a glob pattern.

        Args:
            pattern: Glob pattern to match files (e.g., "*.py", "src/**/*.java")
            repo: The Repo instance to search in
            commit: The commit hash to get files at (default: HEAD)

        Yields:
            TrackedFile instances for each matching file
        """
        # Get list of files matching the pattern at the given commit
        stdout = git(
            "-C",
            str(repo.path),
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            cache_key=f"ls-tree:{repo.repo_id}:{commit}",
        )

        # Filter files matching the glob pattern
        from fnmatch import fnmatch

        for file_path in stdout.splitlines():
            if fnmatch(file_path, pattern):
                yield TrackedFile.get(file_path, repo, commit)

    @staticmethod
    def get(file_path: str, repo: "Repo", commit: str = "HEAD") -> "TrackedFile":
        """
        Get a tracked file with blame information at a specific commit.

        Args:
            file_path: Path to the file (relative to repository root)
            repo: The Repo instance
            commit: The commit hash to get the file at (default: HEAD)

        Returns:
            TrackedFile instance containing the path and list of (Commit, line_content) tuples

        Raises:
            CommitNotFoundError: If the commit does not exist
            FileNotFoundInRepoError: If the file does not exist at the given commit
        """
        # Resolve commit to full hash for consistent caching
        try:
            resolved_commit = git(
                "-C",
                str(repo.path),
                "rev-parse",
                commit,
                cache_key=f"rev-parse:{repo.repo_id}:{commit}",
            ).strip()
        except subprocess.CalledProcessError:
            raise CommitNotFoundError(commit, repo.path)

        try:
            stdout = git(
                "-C",
                str(repo.path),
                "blame",
                "--porcelain",
                "-M",
                "-C",
                "-C",
                resolved_commit,
                "--",
                file_path,
                cache_key=f"blame:{resolved_commit}:{file_path}",
            )
        except subprocess.CalledProcessError as e:
            stderr = (
                e.stderr.decode("utf-8", errors="replace")
                if isinstance(e.stderr, bytes)
                else e.stderr
            )
            if "no such path" in stderr.lower() or "fatal:" in stderr.lower():
                raise FileNotFoundInRepoError(file_path, resolved_commit, repo.path)
            raise

        lines: list[tuple[Commit, str]] = []
        current_hash: str | None = None
        commit_cache: dict[str, Commit] = {}

        for line in stdout.splitlines():
            # Lines starting with a commit hash (40 hex chars) followed by line numbers
            if len(line) >= 40 and all(c in "0123456789abcdef" for c in line[:40]):
                current_hash = line[:40]
            # Lines starting with a tab contain the actual file content
            elif line.startswith("\t"):
                if current_hash is not None:
                    content = line[1:]  # Remove the leading tab
                    # Cache commits to avoid repeated git show calls
                    if current_hash not in commit_cache:
                        commit_cache[current_hash] = repo.get_commit(current_hash)
                    lines.append((commit_cache[current_hash], content))

        return TrackedFile(path=file_path, lines=lines)


class Executable:
    """
    Represents an executable command with caching support.

    The instance is callable and returns cached stdout when possible.
    """

    def __init__(self, command: str) -> None:
        """
        Initialize an Executable instance.

        Args:
            command: The command name or path to the executable

        Raises:
            ExecutableNotFoundError: If the command cannot be found
        """
        # Find the command in PATH
        path = shutil.which(command)
        if path is None:
            raise ExecutableNotFoundError(command)
        self.path: str = path
        self.command = command

    def __repr__(self) -> str:
        return f"Executable({self.command!r}, path={self.path!r})"

    def __call__(self, *args: str, cache_key: str) -> str:
        """
        Execute the command with the given arguments.

        Args:
            *args: Arguments to pass to the command
            cache_key: Key to use for caching the result

        Returns:
            The stdout from the command (cached or fresh)
        """
        global _cache, _cache_dirty

        # Check cache first
        if cache_key in _cache:
            return _cache[cache_key]

        # Execute the command
        exec_args = [self.path] + list(args)
        result = subprocess.run(
            exec_args,
            capture_output=True,
            check=True,
        )

        # Decode stdout with replacement for binary content
        stdout: str = result.stdout.decode("utf-8", errors="replace")
        _cache[cache_key] = stdout
        _cache_dirty = True

        return stdout


# Global git executable
git = Executable("git")


class Repo:
    """
    Represents a Git/GitHub repository.

    Combines local Git operations with GitHub API access.
    """

    _students_config_path: str | None
    _repo_id: str | None

    def __init__(self, path: str | Path) -> None:
        """
        Initialize a Repo instance.

        Args:
            path: Path to the git repository

        Raises:
            ValueError: If the path is not a valid git repository
        """
        self.path = Path(path).resolve()
        self._students_config_path = None
        self._repo_id = None

        # Verify this is a valid git repository
        if not self.path.exists():
            raise ValueError(f"Path does not exist: {self.path}")

        try:
            subprocess.run(
                ["git", "-C", str(self.path), "rev-parse", "--git-dir"],
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Not a valid git repository: {self.path}") from e

    def __repr__(self) -> str:
        return f"Repo({self.path})"

    @property
    def repo_id(self) -> str:
        """
        Get a unique identifier for this repository.

        Uses the remote origin URL to identify the repo, which is unique
        across different forks/clones from the same template.

        Returns:
            A sanitized version of the remote URL (or path if no remote)
        """
        if self._repo_id is None:
            try:
                result = subprocess.run(
                    ["git", "-C", str(self.path), "remote", "get-url", "origin"],
                    capture_output=True,
                    check=True,
                )
                remote_url = result.stdout.decode("utf-8").strip()
                # Extract repo identifier from URL (e.g., "org/repo" from git@github.com:org/repo.git)
                if ":" in remote_url and "@" in remote_url:
                    # SSH format: git@github.com:org/repo.git
                    self._repo_id = remote_url.split(":")[-1].replace(".git", "")
                elif remote_url.startswith("https://"):
                    # HTTPS format: https://github.com/org/repo.git
                    self._repo_id = "/".join(remote_url.split("/")[-2:]).replace(
                        ".git", ""
                    )
                else:
                    self._repo_id = remote_url
            except subprocess.CalledProcessError:
                # No remote, fall back to path basename
                self._repo_id = self.path.name
        return self._repo_id

    def get_commit(self, commit_hash: str) -> Commit:
        """
        Get a Commit instance for the given hash.

        Args:
            commit_hash: The commit hash (full or short)

        Returns:
            Commit instance with hash and message lines

        Raises:
            CommitNotFoundError: If the commit does not exist
        """
        return Commit.get(commit_hash, self)

    def get_tracked_file(self, file_path: str, commit: str = "HEAD") -> TrackedFile:
        """
        Get a tracked file with blame information at a specific commit.

        Args:
            file_path: Path to the file (relative to repository root)
            commit: The commit hash to get the file at (default: HEAD)

        Returns:
            TrackedFile instance containing the path and list of (Commit, line_content) tuples

        Raises:
            CommitNotFoundError: If the commit does not exist
            FileNotFoundInRepoError: If the file does not exist at the given commit
        """
        return TrackedFile.get(file_path, self, commit)

    def files(self, pattern: str, commit: str = "HEAD") -> Iterator[TrackedFile]:
        """
        Yield TrackedFile instances matching a glob pattern.

        Args:
            pattern: Glob pattern to match files (e.g., "*.py", "src/**/*.java")
            commit: The commit hash to get files at (default: HEAD)

        Yields:
            TrackedFile instances for each matching file
        """
        return TrackedFile.files(pattern, self, commit)

    def get_all_commits(self) -> list[tuple[str, list[str]]]:
        """
        Get all commits in the repository with their changed files.

        Returns:
            List of tuples (commit_hash, list of file paths changed in that commit)
        """
        # Get all commits with their files using git log
        # Format: commit hash followed by file list, separated by null chars
        stdout = git(
            "-C",
            str(self.path),
            "log",
            "--pretty=format:%H",
            "--name-only",
            cache_key=f"log-files:{self.repo_id}",
        )

        commits: list[tuple[str, list[str]]] = []
        current_hash: str | None = None
        current_files: list[str] = []

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            # Check if this is a commit hash (40 hex chars)
            if len(line) == 40 and all(c in "0123456789abcdef" for c in line):
                # Save previous commit if exists
                if current_hash is not None:
                    commits.append((current_hash, current_files))
                current_hash = line
                current_files = []
            else:
                # This is a file path
                current_files.append(line)

        # Don't forget the last commit
        if current_hash is not None:
            commits.append((current_hash, current_files))

        return commits

    def get_commit_non_whitespace_files(self, commit_hash: str) -> set[str]:
        """
        Get files that have non-whitespace changes in a commit.

        Parses the unified diff to find files where added or removed
        lines contain non-whitespace content.

        Args:
            commit_hash: The commit hash to check

        Returns:
            Set of file paths with non-whitespace changes
        """
        try:
            # Get the diff for this commit
            diff_output = git(
                "-C",
                str(self.path),
                "show",
                commit_hash,
                "--format=",  # No commit info, just diff
                "-p",  # Show patch
                cache_key=f"show-diff:{self.repo_id}:{commit_hash}",
            )
        except Exception:
            # If we can't get the diff, assume all files have changes
            return set()

        files_with_changes: set[str] = set()
        current_file: str | None = None

        for line in diff_output.splitlines():
            # Detect file header: diff --git a/path b/path
            if line.startswith("diff --git "):
                parts = line.split(" ")
                if len(parts) >= 4:
                    b_path = parts[3]
                    if b_path.startswith("b/"):
                        current_file = b_path[2:]
                    else:
                        current_file = b_path
            # Check added/removed lines for non-whitespace content
            elif current_file and (line.startswith("+") or line.startswith("-")):
                if line.startswith("+++") or line.startswith("---"):
                    continue
                content = line[1:]
                if content.strip():
                    files_with_changes.add(current_file)

        return files_with_changes

    def get_students_json_dict(self) -> dict[str, Any] | None:
        """
        Read the students config file and return it as a dict.

        The config file is identified by the magic ID 619a9605-0e2b-45ee-ac51-a539c59d70bb.
        The method caches the found config file path and re-reads it each time
        (not cached) to always get the most recent version.

        Returns:
            The students config as a dict, or None if not found
        """
        magic_id = "619a9605-0e2b-45ee-ac51-a539c59d70bb"

        # Check if we have a cached path that still works
        if self._students_config_path:
            try:
                config_file = self.path / self._students_config_path
                if config_file.exists():
                    with open(config_file, "r", encoding="utf-8") as f:
                        cached_data: dict[str, Any] = json.load(f)
                    if cached_data.get("id") == magic_id:
                        return cached_data
            except (json.JSONDecodeError, OSError):
                pass
            # Cached path no longer valid, clear it
            self._students_config_path = None

        # Search through all .json files in repo directory (including hidden ones)
        for config_file in self.path.glob("*.json"):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data: dict[str, Any] = json.load(f)

                if data.get("id") == magic_id:
                    # Found it, cache the path
                    self._students_config_path = str(config_file.relative_to(self.path))
                    return data

            except (json.JSONDecodeError, OSError, TypeError):
                # Not a valid JSON file or not a dict, skip it
                continue

        # Also check hidden .json files
        for config_file in self.path.glob(".*.json"):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data2: dict[str, Any] = json.load(f)

                if data2.get("id") == magic_id:
                    # Found it, cache the path
                    self._students_config_path = str(config_file.relative_to(self.path))
                    return data2

            except (json.JSONDecodeError, OSError, TypeError):
                # Not a valid JSON file or not a dict, skip it
                continue

        return None

    def get_students_config(self) -> StudentsConfig:
        """
        Get the students config for this repository.

        Returns:
            StudentsConfig instance

        Raises:
            StudentsConfigError: If the config file cannot be found or is invalid
        """
        data = self.get_students_json_dict()
        if data is None:
            raise StudentsConfigError(
                f"Students config file not found in repository: {self.path}"
            )
        return StudentsConfig.from_dict(data)

    def _get_pr_commits(self, gh_path: str, pr_number: int) -> list[PullRequestCommit]:
        """
        Get commits for a single PR using local git when possible.

        Fetches commit hashes from GitHub, then uses local git to get
        commit details including co-authors from the message body.

        Args:
            gh_path: Path to gh executable
            pr_number: The PR number to fetch commits for

        Returns:
            List of PullRequestCommit instances
        """
        commits: list[PullRequestCommit] = []

        try:
            # Get commit hashes from GitHub (lightweight query)
            result = subprocess.run(
                [
                    gh_path,
                    "pr",
                    "view",
                    str(pr_number),
                    "--json",
                    "commits",
                ],
                cwd=self.path,
                capture_output=True,
                check=True,
            )
            data = json.loads(result.stdout.decode("utf-8"))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            # If we can't fetch commits, return empty list
            return commits

        for commit_data in data.get("commits", []):
            commit_hash = commit_data.get("oid", "")
            if not commit_hash:
                continue

            # Try to get commit details from local git first
            try:
                local_commit = self.get_commit(commit_hash)
                # Use local git data
                co_author_objects = local_commit.get_co_authors()
                co_author_names = [
                    ca.github_username or ca.name for ca in co_author_objects
                ]
                # First message line is the subject
                subject = (
                    local_commit.message_lines[0] if local_commit.message_lines else ""
                )
                commits.append(
                    PullRequestCommit(
                        hash=commit_hash,
                        author_username=local_commit.author.github_username or "",
                        author_email=local_commit.author.email,
                        message=subject,
                        date=local_commit.date,
                        co_authors=co_author_names,
                    )
                )
            except (
                subprocess.CalledProcessError,
                IndexError,
                ValueError,
                UnknownAuthorError,
            ):
                # Fall back to GitHub data if local git fails
                authors = commit_data.get("authors", [])
                if authors:
                    primary_author = authors[0]
                    author_username = primary_author.get("login", "")
                    author_email = primary_author.get("email", "")
                    co_authors = [
                        a.get("login", "") for a in authors[1:] if a.get("login")
                    ]
                else:
                    author_username = ""
                    author_email = ""
                    co_authors = []

                commits.append(
                    PullRequestCommit(
                        hash=commit_hash,
                        author_username=author_username,
                        author_email=author_email,
                        message=commit_data.get("messageHeadline", ""),
                        date=(
                            commit_data.get("authoredDate", "")[:10]
                            if commit_data.get("authoredDate")
                            else ""
                        ),
                        co_authors=co_authors,
                    )
                )

        return commits

    def get_pull_requests(self) -> list[PullRequest]:
        """
        Get all pull requests from the GitHub repository.

        Uses the gh CLI to fetch PR data. Closed/merged PRs are cached
        to reduce API calls. Commit details are fetched individually
        to avoid API limits, using local git when available.

        Returns:
            List of PullRequest instances

        Raises:
            GitHubApiError: If the gh command fails
        """
        global _pr_cache, _cache_dirty

        # Check if gh is available
        gh_path = shutil.which("gh")
        if not gh_path:
            raise ExecutableNotFoundError("gh")

        repo_key = self.repo_id

        try:
            # First, fetch just PR numbers and states to see what's new/changed
            result = subprocess.run(
                [
                    gh_path,
                    "pr",
                    "list",
                    "--state",
                    "all",
                    "--limit",
                    "500",
                    "--json",
                    "number,state",
                ],
                cwd=self.path,
                capture_output=True,
                check=True,
            )
            pr_summary = json.loads(result.stdout.decode("utf-8"))
        except subprocess.CalledProcessError as e:
            stderr = (
                e.stderr.decode("utf-8", errors="replace")
                if isinstance(e.stderr, bytes)
                else str(e.stderr)
            )
            raise GitHubApiError(f"Failed to fetch pull requests: {stderr}") from e
        except json.JSONDecodeError as e:
            raise GitHubApiError(f"Invalid JSON from gh: {e}") from e

        pull_requests: list[PullRequest] = []

        # Determine which PRs need full fetch
        prs_to_fetch: list[int] = []
        for pr_info in pr_summary:
            pr_number = pr_info.get("number", 0)
            state = pr_info.get("state", "OPEN").upper()

            # If it's closed/merged and we have it cached, use cache
            if state != "OPEN" and (repo_key, pr_number) in _pr_cache:
                pull_requests.append(_pr_cache[(repo_key, pr_number)])
            else:
                prs_to_fetch.append(pr_number)

        # Fetch full details for PRs that aren't cached
        if prs_to_fetch:
            fetched_prs = self._fetch_pr_details(gh_path, prs_to_fetch)
            for pr in fetched_prs:
                pull_requests.append(pr)
                # Cache closed PRs
                if not pr.is_open:
                    _pr_cache[(repo_key, pr.number)] = pr
                    _cache_dirty = True

        return pull_requests

    def _fetch_pr_details(
        self, gh_path: str, pr_numbers: list[int]
    ) -> list[PullRequest]:
        """
        Fetch full details for specific PRs.

        Args:
            gh_path: Path to gh executable
            pr_numbers: List of PR numbers to fetch

        Returns:
            List of PullRequest instances
        """
        if not pr_numbers:
            return []

        # Build query for specific PRs
        # We need to fetch them one by one or use search
        # For simplicity, fetch all and filter
        try:
            result = subprocess.run(
                [
                    gh_path,
                    "pr",
                    "list",
                    "--state",
                    "all",
                    "--limit",
                    "500",
                    "--json",
                    "number,author,createdAt,state,mergedAt,mergedBy,reviews,files",
                ],
                cwd=self.path,
                capture_output=True,
                check=True,
            )
            data = json.loads(result.stdout.decode("utf-8"))
        except subprocess.CalledProcessError as e:
            stderr = (
                e.stderr.decode("utf-8", errors="replace")
                if isinstance(e.stderr, bytes)
                else str(e.stderr)
            )
            raise GitHubApiError(f"Failed to fetch pull requests: {stderr}") from e
        except json.JSONDecodeError as e:
            raise GitHubApiError(f"Invalid JSON from gh: {e}") from e

        # Filter to only requested PRs
        pr_number_set = set(pr_numbers)
        pull_requests: list[PullRequest] = []

        for pr_data in data:
            pr_number = pr_data.get("number", 0)
            if pr_number not in pr_number_set:
                continue

            # Fetch commit hashes for this PR individually
            commits = self._get_pr_commits(gh_path, pr_number)

            # Parse reviews
            reviews: list[PullRequestReview] = []
            for review_data in pr_data.get("reviews", []):
                author = review_data.get("author", {})
                reviews.append(
                    PullRequestReview(
                        reviewer_username=author.get("login", ""),
                        state=review_data.get("state", ""),
                        submitted_at=(
                            review_data.get("submittedAt", "")[:10]
                            if review_data.get("submittedAt")
                            else ""
                        ),
                    )
                )

            # Parse author
            author_data = pr_data.get("author", {})
            author_username = author_data.get("login", "")

            # Parse merged by
            merged_by_data = pr_data.get("mergedBy")
            merged_by_username = (
                merged_by_data.get("login", "") if merged_by_data else None
            )

            # Get files with non-whitespace changes from diff
            non_ws_files = self._get_pr_diff_non_whitespace_files(gh_path, pr_number)

            # Parse files with additions/deletions
            files_data = pr_data.get("files", [])
            files = [
                PullRequestFile(
                    path=f.get("path", ""),
                    additions=f.get("additions", 0),
                    deletions=f.get("deletions", 0),
                    has_non_whitespace_changes=f.get("path", "") in non_ws_files,
                )
                for f in files_data
                if f.get("path")
            ]

            pull_requests.append(
                PullRequest(
                    number=pr_number,
                    author_username=author_username,
                    created_at=(
                        pr_data.get("createdAt", "")[:10]
                        if pr_data.get("createdAt")
                        else ""
                    ),
                    state=pr_data.get("state", "OPEN").upper(),
                    merged_at=(
                        pr_data.get("mergedAt", "")[:10]
                        if pr_data.get("mergedAt")
                        else None
                    ),
                    merged_by_username=merged_by_username,
                    commits=commits,
                    reviews=reviews,
                    files=files,
                )
            )

        return pull_requests

    def _get_pr_diff_non_whitespace_files(
        self, gh_path: str, pr_number: int
    ) -> set[str]:
        """
        Get the set of file paths that have non-whitespace changes in a PR.

        Parses the unified diff output to find files where added or removed
        lines contain non-whitespace content.

        Args:
            gh_path: Path to gh executable
            pr_number: PR number to get diff for

        Returns:
            Set of file paths with non-whitespace changes
        """
        try:
            result = subprocess.run(
                [gh_path, "pr", "diff", str(pr_number)],
                cwd=self.path,
                capture_output=True,
                check=True,
            )
            diff_output = result.stdout.decode("utf-8", errors="replace")
        except subprocess.CalledProcessError:
            # If diff fails, assume all files have changes (conservative)
            return set()
        except Exception:
            return set()

        files_with_changes: set[str] = set()
        current_file: str | None = None

        for line in diff_output.splitlines():
            # Detect file header: diff --git a/path b/path
            if line.startswith("diff --git "):
                # Extract path from "diff --git a/path b/path"
                parts = line.split(" ")
                if len(parts) >= 4:
                    # b/path is the destination file
                    b_path = parts[3]
                    if b_path.startswith("b/"):
                        current_file = b_path[2:]  # Remove "b/" prefix
                    else:
                        current_file = b_path
            # Check added/removed lines for non-whitespace content
            elif current_file and (line.startswith("+") or line.startswith("-")):
                # Skip diff headers like +++ and ---
                if line.startswith("+++") or line.startswith("---"):
                    continue
                # Get the actual content (without the +/- prefix)
                content = line[1:]
                # Check if it has non-whitespace content
                if content.strip():
                    files_with_changes.add(current_file)

        return files_with_changes


def get_cache_dir() -> Path:
    """
    Get the appropriate cache directory for the current platform.

    Returns:
        Path to the cache directory
    """
    system = platform.system()

    if system == "Windows":
        # Use LOCALAPPDATA on Windows
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        cache_dir = base / "datool" / "cache"
    elif system == "Darwin":
        # Use ~/Library/Caches on macOS
        cache_dir = Path.home() / "Library" / "Caches" / "datool"
    else:
        # Use XDG_CACHE_HOME on Linux and other Unix-like systems
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache_dir = base / "datool"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_data_dir() -> Path:
    """
    Get the appropriate data directory for the current platform.

    Returns:
        Path to the data directory
    """
    system = platform.system()

    if system == "Windows":
        # Use LOCALAPPDATA on Windows
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        data_dir = base / "datool"
    elif system == "Darwin":
        # Use ~/Library/Application Support on macOS
        data_dir = Path.home() / "Library" / "Application Support" / "datool"
    else:
        # Use XDG_DATA_HOME on Linux and other Unix-like systems
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        data_dir = base / "datool"

    return data_dir


def get_templates_dir() -> Path:
    """
    Get the path to the user templates directory.

    Returns:
        Path to the templates directory (may not exist yet)
    """
    return get_data_dir() / "templates"


def get_cache_file_path() -> Path:
    """Get the path to the cache file."""
    return get_cache_dir() / "cache.pickle"


def load_cache() -> None:
    """Load the entire cache from disk."""
    global _cache, _pr_cache
    cache_path = get_cache_file_path()
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                loaded: Any = pickle.load(f)
                # Handle old cache format (just dict) vs new format (tuple)
                if (
                    isinstance(loaded, tuple)
                    and len(cast(tuple[Any, ...], loaded)) == 2
                ):
                    loaded_tuple = cast(tuple[Any, Any], loaded)
                    _cache = cast(dict[str, str], loaded_tuple[0])
                    _pr_cache = cast(
                        dict[tuple[str, int], "PullRequest"], loaded_tuple[1]
                    )
                elif isinstance(loaded, dict):
                    _cache = cast(dict[str, str], loaded)
                    _pr_cache = {}
                else:
                    _cache = {}
                    _pr_cache = {}
        except (pickle.PickleError, EOFError, OSError):
            # Cache corrupted, start fresh
            _cache = {}
            _pr_cache = {}


def save_cache() -> None:
    """Save the entire cache to disk."""
    global _cache_dirty
    if not _cache_dirty:
        return
    cache_path = get_cache_file_path()
    try:
        with open(cache_path, "wb") as f:
            pickle.dump((_cache, _pr_cache), f)
        _cache_dirty = False
    except OSError:
        # Failed to write cache, ignore
        pass


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle signals by saving cache and exiting."""
    save_cache()
    sys.exit(0)


def setup_cache_handlers() -> None:
    """Set up atexit and signal handlers for cache persistence."""
    atexit.register(save_cache)
    # Handle common termination signals
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _signal_handler)


def clear_cache() -> None:
    """Clear all cached data."""
    global _cache, _pr_cache, _cache_dirty
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Cache cleared: {cache_dir}")
    else:
        print(f"Cache directory does not exist: {cache_dir}")
    _cache = {}
    _pr_cache = {}
    _cache_dirty = False


# Type aliases for analysis results
LinesData = dict[Author, dict[str, list[str]]]
CommitsData = dict[Author, set[str]]
GitHubStatsData = dict[Author, GitHubStats]


@dataclass
class AnalysisResult:
    """Results from analyzing a repository."""

    alone_lines: LinesData
    collab_lines: LinesData
    alone_commits: CommitsData
    collab_commits: CommitsData
    file_commits: dict[str, set[str]]
    github_stats: GitHubStatsData | None = None


def _collect_github_stats(
    repo: Repo,
    students_config: StudentsConfig,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> GitHubStatsData:
    """
    Collect GitHub PR statistics for each student.

    Only PRs that contain files matching include patterns (and not matching
    exclude patterns) are counted.

    PR.Alone: PRs where the author did all commits without co-authors.
    PR.Collab: PRs where any commit has co-authors.

    Args:
        repo: The repository to analyze
        students_config: The students configuration
        include_patterns: Glob patterns for files to include
        exclude_patterns: Glob patterns for files to exclude

    Returns:
        Dictionary mapping Author to their GitHubStats
    """
    from fnmatch import fnmatch

    # Build mapping from GitHub username to Author
    username_to_author: dict[str, Author] = {}
    for student in students_config.students:
        username_to_author[student.github_username.lower()] = student

    # Initialize stats for each student
    stats: GitHubStatsData = {s: GitHubStats() for s in students_config.students}

    # Helper to check if a file matches include patterns
    def matches_include(file_path: str) -> bool:
        for pattern in include_patterns:
            if fnmatch(file_path, pattern):
                return True
        return False

    # Helper to check if a file should be excluded
    def should_exclude(file_path: str) -> bool:
        for pattern in exclude_patterns:
            if fnmatch(file_path, pattern):
                return True
        return False

    # Helper to check if PR has matching files with non-whitespace changes
    def pr_has_matching_files(pr: PullRequest) -> bool:
        for pr_file in pr.files:
            if matches_include(pr_file.path) and not should_exclude(pr_file.path):
                if pr_file.has_non_whitespace_changes:
                    return True
        return False

    # Helper to check if PR is collaborative (has any commits with co-authors)
    def pr_is_collab(pr: PullRequest) -> bool:
        for commit in pr.commits:
            if commit.co_authors:
                return True
        return False

    # Fetch PRs
    pull_requests = repo.get_pull_requests()

    for pr in pull_requests:
        # Skip bot authors
        if pr.author_username.startswith("app/"):
            continue

        # Skip PRs that don't have matching files with non-empty line changes
        if not pr_has_matching_files(pr):
            continue

        # Skip PRs that were closed without being merged
        if not pr.is_merged and not pr.is_open:
            continue

        # Find author
        author = username_to_author.get(pr.author_username.lower())
        if author:
            is_collab = pr_is_collab(pr)

            if is_collab:
                if pr.is_merged:
                    stats[author].prs_collab_merged += 1
                else:
                    stats[author].prs_collab_open += 1
            else:
                if pr.is_merged:
                    stats[author].prs_alone_merged += 1
                else:
                    stats[author].prs_alone_open += 1

        # Count reviews (approvals and change requests)
        # Only count one approval/change-request per reviewer per PR
        seen_approvers: set[Author] = set()
        seen_change_requesters: set[Author] = set()
        for review in pr.reviews:
            reviewer = username_to_author.get(review.reviewer_username.lower())
            if reviewer:
                if review.state == "APPROVED":
                    if reviewer not in seen_approvers:
                        stats[reviewer].approvals_given += 1
                        seen_approvers.add(reviewer)
                elif review.state == "CHANGES_REQUESTED":
                    if reviewer not in seen_change_requesters:
                        stats[reviewer].change_requests_given += 1
                        seen_change_requesters.add(reviewer)

    return stats


def _collect_line_stats(
    repo: Repo,
    students_config: StudentsConfig,
    ignored_authors: set[Author],
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> tuple[LinesData, LinesData, dict[str, set[str]], dict[str, list[Author]]]:
    """Collect line statistics from blame data."""
    from fnmatch import fnmatch

    alone_lines: LinesData = {s: {} for s in students_config.students}
    collab_lines: LinesData = {s: {} for s in students_config.students}
    file_commits: dict[str, set[str]] = {}
    coauthors_cache: dict[str, list[Author]] = {}

    def should_exclude(file_path: str) -> bool:
        for pattern in exclude_patterns:
            if fnmatch(file_path, pattern):
                return True
        return False

    for pattern in include_patterns:
        for tracked_file in repo.files(pattern):
            if should_exclude(tracked_file.path):
                continue

            for commit, content in tracked_file.lines:
                if not content.strip():
                    continue
                if commit.author in ignored_authors:
                    continue

                if tracked_file.path not in file_commits:
                    file_commits[tracked_file.path] = set()
                file_commits[tracked_file.path].add(commit.hash)

                if commit.hash not in coauthors_cache:
                    coauthors_cache[commit.hash] = commit.get_co_authors()
                co_authors = coauthors_cache[commit.hash]
                valid_co_authors = [
                    ca for ca in co_authors if ca not in ignored_authors
                ]
                is_collab = len(valid_co_authors) > 0

                if is_collab:
                    if commit.author in collab_lines:
                        if tracked_file.path not in collab_lines[commit.author]:
                            collab_lines[commit.author][tracked_file.path] = []
                        collab_lines[commit.author][tracked_file.path].append(content)

                    for co_author in valid_co_authors:
                        if co_author in collab_lines:
                            if tracked_file.path not in collab_lines[co_author]:
                                collab_lines[co_author][tracked_file.path] = []
                            collab_lines[co_author][tracked_file.path].append(content)
                else:
                    if commit.author in alone_lines:
                        if tracked_file.path not in alone_lines[commit.author]:
                            alone_lines[commit.author][tracked_file.path] = []
                        alone_lines[commit.author][tracked_file.path].append(content)

    return alone_lines, collab_lines, file_commits, coauthors_cache


def _collect_commit_stats(
    repo: Repo,
    students_config: StudentsConfig,
    ignored_authors: set[Author],
    include_patterns: list[str],
    exclude_patterns: list[str],
    coauthors_cache: dict[str, list[Author]],
) -> tuple[CommitsData, CommitsData]:
    """Collect commit statistics."""
    from fnmatch import fnmatch

    alone_commits: CommitsData = {s: set() for s in students_config.students}
    collab_commits: CommitsData = {s: set() for s in students_config.students}

    def matches_include(file_path: str) -> bool:
        for pattern in include_patterns:
            if fnmatch(file_path, pattern):
                return True
        return False

    def should_exclude(file_path: str) -> bool:
        for pattern in exclude_patterns:
            if fnmatch(file_path, pattern):
                return True
        return False

    for commit_hash, files in repo.get_all_commits():
        matching_files = [
            f for f in files if matches_include(f) and not should_exclude(f)
        ]
        if not matching_files:
            continue

        # Check if commit has non-whitespace changes to any matching file
        non_ws_files = repo.get_commit_non_whitespace_files(commit_hash)
        has_non_ws_changes = any(f in non_ws_files for f in matching_files)
        if not has_non_ws_changes:
            continue

        try:
            commit = repo.get_commit(commit_hash)
        except (CommitNotFoundError, UnknownAuthorError):
            continue

        if commit.author in ignored_authors:
            continue

        if commit_hash not in coauthors_cache:
            coauthors_cache[commit_hash] = commit.get_co_authors()
        co_authors = coauthors_cache[commit_hash]
        valid_co_authors = [ca for ca in co_authors if ca not in ignored_authors]
        is_collab = len(valid_co_authors) > 0

        if is_collab:
            if commit.author in collab_commits:
                collab_commits[commit.author].add(commit_hash)
            for co_author in valid_co_authors:
                if co_author in collab_commits:
                    collab_commits[co_author].add(commit_hash)
        else:
            if commit.author in alone_commits:
                alone_commits[commit.author].add(commit_hash)

    return alone_commits, collab_commits


def _print_summary(
    students_config: StudentsConfig,
    result: AnalysisResult,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> None:
    """Print the student summary table."""
    print("Include:", ", ".join(include_patterns) if include_patterns else "(all)")
    print("Exclude:", ", ".join(exclude_patterns) if exclude_patterns else "(none)")
    print()

    max_name_len = max(len(s.name) for s in students_config.students)
    has_gh = result.github_stats is not None

    print("Student summary:")
    print("  C=Commits, L=Lines, Alone=without co-authors, Collab=with co-authors")

    # Helper to format PR count: "merged(unmerged)" or just "merged" if no unmerged
    def fmt_pr(merged: int, unmerged: int) -> str:
        if unmerged > 0:
            return f"{merged}({unmerged})"
        return str(merged)

    # Build header based on whether we have GitHub stats
    if has_gh:
        header = (
            f"{'ID':<3} {'Name':<{max_name_len}}  "
            f"{'PR.Alone':>9}  {'PR.Collab':>10}  {'PR.Appr':>8}  "
            f"{'C.Alone':>8}  {'C.Collab':>9}  {'L.Alone':>8}  {'L.Collab':>9}"
        )
        sep_len = (
            3
            + 1
            + max_name_len
            + 2
            + 9
            + 2
            + 10
            + 2
            + 8
            + 2
            + 8
            + 2
            + 9
            + 2
            + 8
            + 2
            + 9
        )
    else:
        header = (
            f"{'ID':<3} {'Name':<{max_name_len}}  "
            f"{'C.Alone':>8}  {'C.Collab':>9}  {'L.Alone':>8}  {'L.Collab':>9}"
        )
        sep_len = 3 + 1 + max_name_len + 2 + 8 + 2 + 9 + 2 + 8 + 2 + 9

    print(header)
    print("-" * sep_len)

    for student in students_config.students:
        total_alone = sum(len(lines) for lines in result.alone_lines[student].values())
        total_collab = sum(
            len(lines) for lines in result.collab_lines[student].values()
        )
        c_alone = len(result.alone_commits[student])
        c_collab = len(result.collab_commits[student])

        if has_gh and result.github_stats:
            gh = result.github_stats[student]
            pr_alone = fmt_pr(gh.prs_alone_merged, gh.prs_alone_open)
            pr_collab = fmt_pr(gh.prs_collab_merged, gh.prs_collab_open)
            print(
                f"{student.student_id:<3} {student.name:<{max_name_len}}  "
                f"{pr_alone:>9}  {pr_collab:>10}  {gh.approvals_given:>8}  "
                f"{c_alone:>8}  {c_collab:>9}  {total_alone:>8}  {total_collab:>9}"
            )
        else:
            print(
                f"{student.student_id:<3} {student.name:<{max_name_len}}  "
                f"{c_alone:>8}  {c_collab:>9}  {total_alone:>8}  {total_collab:>9}"
            )


def _print_file_details(
    repo: Repo,
    students_config: StudentsConfig,
    result: AnalysisResult,
) -> None:
    """Print the file details table and commits lookup."""
    print()
    print("File details:")
    print("  A=Alone (no co-authors), C=Collab (with co-authors), number=line count")

    student_ids = [s.student_id for s in students_config.students]
    col_width = 8

    header_parts: list[str] = []
    for sid in student_ids:
        header_parts.append(f"{sid:^{col_width}}")

    all_files = sorted(result.file_commits.keys())

    max_filename_len = 4
    for file_path in all_files:
        if "/" in file_path:
            filename = file_path.rsplit("/", 1)[1]
        else:
            filename = file_path
        max_filename_len = max(max_filename_len, len(filename))

    header_parts.extend([f"{'File':<{max_filename_len}}", "Commit"])
    print(" ".join(header_parts))
    print("-" * 80)

    current_dir: str | None = None
    all_commit_hashes: set[str] = set()

    for file_path in all_files:
        if "/" in file_path:
            dir_path = file_path.rsplit("/", 1)[0] + "/"
            filename = file_path.rsplit("/", 1)[1]
        else:
            dir_path = ""
            filename = file_path

        if dir_path != current_dir:
            current_dir = dir_path
            if dir_path:
                print(f"\n{dir_path}")

        commits_for_file = sorted(result.file_commits[file_path])
        all_commit_hashes.update(commits_for_file)

        row_parts: list[str] = []
        for student in students_config.students:
            a_count = len(result.alone_lines[student].get(file_path, []))
            c_count = len(result.collab_lines[student].get(file_path, []))
            parts: list[str] = []
            if a_count > 0:
                parts.append(f"A{a_count}")
            if c_count > 0:
                parts.append(f"C{c_count}")
            cell = ",".join(parts)
            row_parts.append(f"{cell:^{col_width}}")

        # First row: student stats + filename + first commit
        first_commit = commits_for_file[0][:8] if commits_for_file else ""
        row_parts.extend([f"{filename:<{max_filename_len}}", first_commit])
        print(" ".join(row_parts))

        # Additional rows: blank cells + blank filename + remaining commits
        blank_cells = " ".join([" " * col_width] * len(students_config.students))
        blank_filename = " " * max_filename_len
        for commit_hash in commits_for_file[1:]:
            print(f"{blank_cells} {blank_filename} {commit_hash[:8]}")

    # Print commits lookup table
    print()
    print("Commits:")
    print(f"{'Hash':<10} {'Date':<12} Message")
    print("-" * 80)

    commit_objects = [repo.get_commit(h) for h in all_commit_hashes]
    commit_objects.sort(key=lambda c: c.date, reverse=True)

    for commit in commit_objects:
        msg = commit.message_lines[0] if commit.message_lines else ""
        if len(msg) > 50:
            msg = msg[:50] + "..."
        print(f"{commit.hash[:8]:<10} {commit.date:<12} {msg}")


def init_repo(repo_path: str) -> None:
    """
    Initialize a repository with GitHub Actions workflow and datool script.

    Two-phase initialization:
    1. First run: Creates editable templates in user data directory and exits.
       User should customize templates for their course before continuing.
    2. Second run: Copies templates + datool.py into the target repository.

    Args:
        repo_path: Path to the target repository

    Raises:
        SystemExit: If templates don't exist (first run) or target files exist
    """
    # Phase 1: Check if user templates exist FIRST (before validating target)
    user_templates_dir = get_templates_dir()

    if not user_templates_dir.exists():
        # First run: initialize templates from bundled defaults and exit
        _init_user_templates(user_templates_dir)
        sys.exit(0)

    # Phase 2: Templates exist, now validate target and copy files
    target = Path(repo_path).resolve()

    if not target.exists():
        print(f"Error: Path does not exist: {target}", file=sys.stderr)
        sys.exit(1)

    # Templates exist: copy them to the target repo along with datool.py
    script_path = Path(__file__).resolve()

    # Files to copy from user templates
    template_files = [
        (
            ".github/workflows/participation.yml",
            user_templates_dir / ".github" / "workflows" / "participation.yml",
        ),
        (".students.json.example", user_templates_dir / ".students.json"),
    ]

    # datool.py comes from the script itself
    all_files = [(".github/scripts/datool.py", script_path)] + template_files

    # Check if any target files already exist
    existing_files: list[str] = []
    for rel_path, _ in all_files:
        target_file = target / rel_path
        if target_file.exists():
            existing_files.append(rel_path)

    if existing_files:
        print("Error: The following files already exist:", file=sys.stderr)
        for f in existing_files:
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)

    # Copy files
    for rel_path, src_file in all_files:
        if not src_file.exists():
            print(
                f"Warning: Source file not found, skipping: {src_file}", file=sys.stderr
            )
            continue

        target_file = target / rel_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, target_file)
        print(f"Created: {rel_path}")

    print()
    print("Initialization complete!")
    print("Next steps:")
    print("  1. Rename .students.json.example to .students.json")
    print("  2. Edit .students.json with your student information")
    print("  3. Commit and push the changes")


def _init_user_templates(user_templates_dir: Path) -> None:
    """
    Initialize user templates directory from bundled defaults.

    Copies template files (workflow, example config) to the user's data directory.
    Does NOT copy datool.py - that's copied directly from the running script.

    Args:
        user_templates_dir: Path to create templates in
    """
    # Find bundled templates relative to this script
    script_path = Path(__file__).resolve()
    bundled_templates = script_path.parent / "templates"

    if not bundled_templates.exists():
        print(
            "Error: Bundled templates not found. This may happen if datool was "
            "installed incorrectly. Please reinstall.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create user templates directory
    user_templates_dir.mkdir(parents=True, exist_ok=True)

    # Copy template files (excluding datool.py)
    files_to_copy = []
    for src_file in bundled_templates.rglob("*"):
        if src_file.is_file():
            rel_path = src_file.relative_to(bundled_templates)
            # Skip datool.py - it's copied from the running script
            if rel_path.name == "datool.py":
                continue
            files_to_copy.append((rel_path, src_file))

    for rel_path, src_file in files_to_copy:
        target_file = user_templates_dir / rel_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, target_file)
        print(f"Created template: {rel_path}")

    print()
    print(f"Templates initialized in: {user_templates_dir}")
    print()
    print("You can now customize the templates for your course:")
    print(f"  - {user_templates_dir / '.github' / 'workflows' / 'participation.yml'}")
    print(f"  - {user_templates_dir / '.students.json'}")
    print()
    print("After customizing, run 'datool --init <repo>' again to initialize a repo.")


def fetch_readme() -> str:
    """
    Fetch the README.md from the GitHub repository.

    Uses a simple cache to avoid repeated network requests.

    Returns:
        The README content as a string

    Raises:
        ConnectionError: If the README cannot be fetched
    """
    global _cache, _cache_dirty

    cache_key = "readme:kc8se/datool"

    # Check cache first
    if cache_key in _cache:
        return _cache[cache_key]

    # Fetch from GitHub
    import urllib.error
    import urllib.request

    url = "https://raw.githubusercontent.com/kc8se/datool/main/README.md"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            content = response.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Could not fetch README. Are you connected to the internet?\n"
            f"Details: {e}"
        ) from e
    except Exception as e:
        raise ConnectionError(
            f"Could not fetch README. Are you connected to the internet?\n"
            f"Details: {e}"
        ) from e

    # Cache the result
    _cache[cache_key] = content
    _cache_dirty = True

    return content


def show_readme() -> None:
    """Fetch and display the README.md from GitHub."""
    load_cache()
    setup_cache_handlers()

    try:
        readme = fetch_readme()
        print(readme)
    except ConnectionError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Gather project participation information using git and gh.",
        epilog=(
            "For full documentation including how all values are calculated, "
            "run: datool --readme"
        ),
    )
    parser.add_argument(
        "repo",
        nargs="?",
        default=".",
        help="Path to the git repository (default: current directory)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached data and exit",
    )
    parser.add_argument(
        "--github",
        action="store_true",
        help="Include GitHub PR statistics (auto-enabled in CI environments)",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize repo with GitHub Actions workflow and datool script",
    )
    parser.add_argument(
        "--readme",
        action="store_true",
        help="Show full documentation from README.md",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="datool 1.0.0",
    )
    args = parser.parse_args()

    if args.readme:
        show_readme()
        sys.exit(0)

    if args.clear_cache:
        clear_cache()
        sys.exit(0)

    if args.init:
        init_repo(args.repo)
        sys.exit(0)

    # Determine if GitHub stats should be collected
    # Auto-enable in CI environments (GitHub Actions sets CI=true)
    use_github = args.github or bool(os.environ.get("CI"))

    load_cache()
    setup_cache_handlers()

    try:
        repo = Repo(args.repo)
        students_config = repo.get_students_config()
        ignored_authors = set(students_config.ignore)

        include_patterns = students_config.include_patterns or ["*"]
        exclude_patterns = students_config.exclude_patterns or []

        # Collect line stats
        alone_lines, collab_lines, file_commits, coauthors_cache = _collect_line_stats(
            repo, students_config, ignored_authors, include_patterns, exclude_patterns
        )

        # Collect commit stats
        alone_commits, collab_commits = _collect_commit_stats(
            repo,
            students_config,
            ignored_authors,
            include_patterns,
            exclude_patterns,
            coauthors_cache,
        )

        # Collect GitHub stats if enabled
        github_stats: GitHubStatsData | None = None
        if use_github:
            github_stats = _collect_github_stats(
                repo, students_config, include_patterns, exclude_patterns
            )

        result = AnalysisResult(
            alone_lines=alone_lines,
            collab_lines=collab_lines,
            alone_commits=alone_commits,
            collab_commits=collab_commits,
            file_commits=file_commits,
            github_stats=github_stats,
        )

        _print_summary(students_config, result, include_patterns, exclude_patterns)
        _print_file_details(repo, students_config, result)

    except (
        ValueError,
        CommitNotFoundError,
        FileNotFoundInRepoError,
        ExecutableNotFoundError,
        StudentsConfigError,
        UnknownAuthorError,
        GitHubApiError,
    ) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        stderr = (
            e.stderr.decode("utf-8", errors="replace")
            if isinstance(e.stderr, bytes)
            else e.stderr
        )
        print(f"Error running git command: {stderr}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

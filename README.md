# datool

Analyze student project participation using git blame and GitHub PRs.

## What it does

- Attributes lines and commits to students based on git blame
- Distinguishes **Alone** (solo work) vs **Collab** (with co-authors via `Co-authored-by:` trailers)
- Optionally fetches GitHub PR statistics
- Outputs a summary table and file-by-file breakdown

## Install

```bash
pipx install git+https://github.com/kc8se/datool.git
```

Or for development:

```bash
git clone https://github.com/kc8se/datool.git
pipx install -e ./datool
```

## Initialize a student repo

The first time you run `--init`, datool creates customizable templates in your user data directory:

```bash
datool --init /path/to/student-repo
# Templates initialized in: ~/.local/share/datool/templates (Linux)
#                           ~/Library/Application Support/datool/templates (macOS)
#                           %LOCALAPPDATA%\datool\templates (Windows)
```

Customize the templates for your course:

- `.students.json` — default student config template
- `.github/workflows/participation.yml` — GitHub Actions workflow

Then run `--init` again to copy them to a student repo:

```bash
datool --init /path/to/student-repo
```

This creates:

- `.github/scripts/datool.py` — analysis script (copied from the installed datool)
- `.github/workflows/participation.yml` — from your templates
- `.students.json.example` — from your templates

Then in the student repo:

1. Rename `.students.json.example` → `.students.json`
2. Edit with student info:

   ```json
   {
     "id": "619a9605-0e2b-45ee-ac51-a539c59d70bb",
     "students": [
       {
         "id": "1",
         "name": "Alice",
         "email": "alice@example.com",
         "github_username": "alice"
       }
     ],
     "ignore": [],
     "files": {
       "include": ["src/**/*.py"],
       "exclude": ["**/test_*.py"]
     }
   }
   ```

3. Commit and push

## Usage

```bash
# Analyze a repo (local git only)
datool /path/to/repo

# Include GitHub PR stats
datool --github /path/to/repo

# Clear cache
datool --clear-cache .
```

In CI (GitHub Actions), `--github` is auto-enabled.

## How columns are calculated

All statistics are filtered by the `files.include` and `files.exclude` glob patterns in `.students.json`. Only files matching include patterns and NOT matching exclude patterns are considered.

Additionally, all statistics require **non-whitespace changes** — whitespace-only lines or changes are ignored.

### File filtering

```json
{
  "files": {
    "include": ["**/*.java"],
    "exclude": ["**/test/**", "**/generated/**"]
  }
}
```

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.

To add the license file, you can use GitHub's "Add file → Create new file" and choose the MIT template, or create a `LICENSE` file locally containing the full MIT text (replace YEAR and COPYRIGHT HOLDER), then commit and push.

- **include**: Glob patterns for files to analyze (e.g., `**/*.java`, `src/**/*.py`)
- **exclude**: Glob patterns to skip (e.g., boilerplate, generated code, tests)

### Column definitions

| Column        | Description                                                                                                                                               |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PR.Alone**  | PRs authored where all commits have no co-authors. Only counts PRs with at least one matching file that has non-whitespace changes.                       |
| **PR.Collab** | PRs authored where at least one commit has `Co-authored-by:` trailers. Same filtering as PR.Alone.                                                        |
| **PR.Appr**   | Number of PRs approved by this student. Only counts one approval per PR (not multiple reviews). Only counts PRs that pass the file/non-whitespace filter. |
| **C.Alone**   | Commits with no co-authors that modify matching files with non-whitespace changes.                                                                        |
| **C.Collab**  | Commits with `Co-authored-by:` trailers that modify matching files with non-whitespace changes. Both the author and co-authors get credit.                |
| **L.Alone**   | Non-whitespace lines attributed to this student (via git blame) from commits without co-authors.                                                          |
| **L.Collab**  | Non-whitespace lines attributed to this student from commits with co-authors. Both author and co-authors get credit for the same lines.                   |

### Non-whitespace detection

- **Lines (L.Alone/L.Collab)**: Blank lines (`line.strip() == ""`) are not counted
- **Commits (C.Alone/C.Collab)**: The commit diff is parsed; only commits with added/removed lines containing non-whitespace content are counted
- **PRs (PR.Alone/PR.Collab/PR.Appr)**: The PR diff is parsed; only PRs with files containing non-whitespace changes are counted

### Git blame behavior

Line attribution uses `git blame -C -C` which:

- Analyzes the current state of files at **HEAD**
- Detects code **moved or copied** between files (not just the commit that last touched the line)
- Attributes lines to the original author who wrote the code, even if it was later moved

This means if Student A writes code in `FileA.java` and Student B moves it to `FileB.java`, Student A still gets credit for those lines.

### Co-authored-by detection

Commits are considered collaborative if they contain `Co-authored-by:` trailers in the commit message.

**Correct format** (GitHub standard):

```text
Co-authored-by: Name <email@example.com>
```

**Forgiving parsing** — the tool also tries to match common mistakes:

```text
Co-authored-by: Name email@example.com    # Missing angle brackets
Co-authored-by Name <email@example.com>   # Missing colon
co-authored-by: Name <email@example.com>  # Lowercase
Co-authored: Name <email@example.com>     # Missing "-by"
Coauthored-by: Name <email@example.com>   # Missing hyphen
```

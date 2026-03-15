# =============================================================================
# FILE: filters.py
# PURPOSE: Filter out junk files from GitIngest's raw repo dump
#
# SINGLE RESPONSIBILITY: this file knows ONLY about filtering.
# It doesn't know about LangChain, FAISS, embeddings, or Groq.
# ingest.py calls it. That's the only connection.
#
# WHY A SEPARATE FILE:
#   ingest.py already handles fetch → split → embed → store.
#   Adding filtering there would give it 4 responsibilities.
#   Keeping filters here means you can improve, test, or swap
#   the filtering logic without ever opening ingest.py.
#   This is the Single Responsibility Principle in practice.
# =============================================================================


# =============================================================================
# BLOCKLISTS — what we want to skip
# =============================================================================
#
# WHY SETS and not LISTS?
#   Checking "is X in collection" is O(1) for a set (hash lookup)
#   and O(n) for a list (scan every item).
#   We check every file against these collections — sets are faster.
#   For 500 files × 20 blocked extensions = 10,000 checks.
#   With a list: 10,000 linear scans. With a set: 10,000 instant lookups.
#   Small difference here, but the right habit to build.
#
# WHY THESE SPECIFIC EXTENSIONS:
#   .lock files     → package-lock.json, yarn.lock, Pipfile.lock
#                     These are auto-generated dependency resolution trees.
#                     50,000+ lines of hashes and version numbers.
#                     Zero value for understanding the codebase.
#
#   .pyc files      → compiled Python bytecode. Binary format stored as text
#                     looks like garbage characters. Actively harmful to embed.
#
#   .min.js/.min.css → minified files. All whitespace and variable names
#                      removed. "function a(b,c){return b+c}" tells you nothing.
#
#   .map files      → source maps for debugging minified code. Machine format.
#
#   image/font files → binary data. When GitIngest reads them as text you get
#                      thousands of garbage characters polluting your chunks.
#
# WHY THESE SPECIFIC DIRECTORIES:
#   node_modules/   → can contain 50,000+ files. None of them are your code.
#                     This single directory is why large JS repos are so slow.
#
#   .git/           → git's internal database. Commit hashes, pack files.
#                     Zero relevance to understanding the codebase.
#
#   dist/ build/    → generated output. The SOURCE is already indexed.
#                     Indexing the compiled output too = duplicate noise.
#
#   __pycache__/    → Python's compiled bytecode cache. Same as .pyc above.
#
#   .venv/ venv/ vendor/ → dependencies. Not your code.
#
#   migrations/     → auto-generated database migration files. Hundreds of
#                     files with schema changes — useful for DB devs but
#                     noise for understanding application logic.
# =============================================================================

BLOCKED_EXTENSIONS = {
    # Dependency lock files
    ".lock",
    # Compiled/bytecode
    ".pyc", ".pyo", ".pyd",
    # Minified files
    ".min.js", ".min.css",
    # Source maps
    ".map",
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".bmp",
    # Fonts
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    # Binary/archive
    ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z",
    # Excel/office
    ".xlsx", ".xls", ".docx", ".pptx",
    # Compiled binaries
    ".exe", ".dll", ".so", ".dylib",
    # Database files
    ".sqlite", ".db",
}

BLOCKED_DIRECTORIES = {
    "node_modules/",
    ".git/",
    "dist/",
    "build/",
    "__pycache__/",
    "migrations/",
    ".venv/",
    "venv/",
    "vendor/",
    ".next/",       # Next.js build output
    ".nuxt/",       # Nuxt.js build output
    "coverage/",    # test coverage reports
    ".pytest_cache/",
    ".mypy_cache/",
    "target/",      # Rust/Java build output
    "out/",         # general build output
}


# =============================================================================
# CONCEPT — State Machine
# =============================================================================
# A state machine is a pattern where your code tracks a "state" variable
# and changes behaviour based on it.
#
# Our parser has exactly TWO states:
#   KEEPING  → we're inside a file section we want to keep
#   SKIPPING → we're inside a file section we want to throw away
#
# The state changes ONLY when we hit a new file header.
# Between headers, we stay in the current state.
#
# Visual example:
#
#   ================================================   ← separator line
#   File: src/auth/utils.py                            ← header → check filename
#   ================================================   ← separator line
#   def authenticate():                                ← state=KEEPING → keep this
#       ...                                            ← state=KEEPING → keep this
#   ================================================   ← separator line
#   File: node_modules/lodash/lodash.js                ← header → check filename → BLOCKED
#   ================================================   ← separator line
#   (function(){var n=1/0...                           ← state=SKIPPING → throw this away
#   ================================================   ← separator line
#   File: src/models/user.py                           ← header → check filename → OK
#   ================================================   ← separator line
#   class User:                                        ← state=KEEPING → keep this
#
# This pattern appears in: parsers, compilers, protocol handlers,
# game logic, UI state management. Worth understanding deeply.
# =============================================================================

def _should_skip_file(filepath: str) -> bool:
    """
    Given a filepath string, returns True if it should be filtered out.

    Checks two things:
      1. Does the file path contain a blocked directory?
      2. Does the file have a blocked extension?

    WHY a helper function?
      The main filter_content() function is already managing state.
      Keeping the "should I skip this?" logic separate makes both
      functions easier to read and test independently.
      You can test _should_skip_file("node_modules/x.js") → True
      without running the whole parser.
    """

    # Check blocked directories
    # We check if ANY blocked directory string appears anywhere in the path.
    # Example: "frontend/node_modules/react/index.js" contains "node_modules/"
    # so it gets caught even when nested deep.
    for blocked_dir in BLOCKED_DIRECTORIES:
        if blocked_dir in filepath:
            return True

    # Check blocked extensions
    # We find the last "." in the filename to get the extension.
    # Why not use os.path.splitext()?
    #   filepath here is a string from GitIngest's output, not a real OS path.
    #   It could be a relative path like "src/utils.min.js".
    #   Finding the extension manually is safer and has no dependencies.
    dot_index = filepath.rfind(".")     # rfind finds the LAST "." in the string
    if dot_index != -1:                 # -1 means no "." found (no extension)
        extension = filepath[dot_index:].lower()   # ".JS" → ".js" (normalize case)
        if extension in BLOCKED_EXTENSIONS:
            return True

    return False


def filter_content(raw_content: str) -> str:
    """
    Parses GitIngest's raw content string and removes junk file sections.

    GitIngest formats content like this:
        ================================================
        File: path/to/file.ext
        ================================================
        ...file contents...
        ================================================
        File: another/file.ext
        ================================================
        ...file contents...

    This function walks through line by line, detects file headers,
    decides keep/skip per file, and returns only the kept sections.

    Returns the filtered content string + prints a summary of what was removed.
    """

    # Split the entire content into individual lines
    # splitlines() handles \n, \r\n, \r — works on all platforms
    lines = raw_content.splitlines()

    kept_lines = []         # lines we're keeping
    current_state = True    # True = KEEPING, False = SKIPPING
    current_file = ""       # tracks which file we're currently in (for logging)

    kept_files = []         # for summary reporting
    skipped_files = []      # for summary reporting

    # ── State machine loop ────────────────────────────────────────────────────
    # We need to look at THREE consecutive lines to detect a file header:
    #   line[i]   = "================================================"
    #   line[i+1] = "File: path/to/file.ext"
    #   line[i+2] = "================================================"
    #
    # So we iterate with an index instead of "for line in lines"
    # because we need to look ahead at the next lines.
    # ─────────────────────────────────────────────────────────────────────────

    i = 0
    while i < len(lines):
        line = lines[i]

        # ── Detect file header ────────────────────────────────────────────────
        # A separator line is all "=" characters (at least 10 of them).
        # We check this first, then look ahead to see if next line is "File: ..."
        # ─────────────────────────────────────────────────────────────────────
        is_separator = line.strip().startswith("=") and len(line.strip()) > 10

        if is_separator and i + 1 < len(lines) and lines[i + 1].startswith("File:"):
            # We found a file header block — extract the filepath
            # "File: src/auth/utils.py" → "src/auth/utils.py"
            filepath = lines[i + 1].replace("File:", "").strip()
            current_file = filepath

            # Ask _should_skip_file() whether to filter this file
            if _should_skip_file(filepath):
                current_state = False   # switch to SKIPPING state
                skipped_files.append(filepath)
            else:
                current_state = True    # switch to KEEPING state
                kept_files.append(filepath)

        # ── Apply current state ───────────────────────────────────────────────
        # Regardless of whether this line is a header or content,
        # we keep it if current_state is True, skip it if False.
        # ─────────────────────────────────────────────────────────────────────
        if current_state:
            kept_lines.append(line)

        i += 1

    # ── Build result and print summary ────────────────────────────────────────
    filtered_content = "\n".join(kept_lines)

    original_chars = len(raw_content)
    filtered_chars = len(filtered_content)
    removed_pct = ((original_chars - filtered_chars) / original_chars * 100) if original_chars > 0 else 0

    print(f"🔍 Filter results:")
    print(f"   Files kept:    {len(kept_files)}")
    print(f"   Files skipped: {len(skipped_files)}")
    print(f"   Content before: {original_chars:,} chars")
    print(f"   Content after:  {filtered_chars:,} chars")
    print(f"   Removed: {removed_pct:.1f}%")

    # Uncomment below to see exactly which files were skipped:
    # for f in skipped_files:
    #     print(f"   SKIPPED: {f}")

    return filtered_content
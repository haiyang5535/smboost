from __future__ import annotations


def shrink_small_model_stdin_problem(problem_text: str, *, max_chars: int) -> str:
    if len(problem_text) <= max_chars:
        return problem_text

    markers = ["Input", "Output", "Sample Input", "Sample Output", "Note"]

    def _section(marker: str) -> str:
        start = problem_text.find(marker)
        if start < 0:
            return ""
        ends = [
            problem_text.find(next_marker, start + len(marker))
            for next_marker in markers
            if next_marker != marker
        ]
        ends = [end for end in ends if end >= 0]
        end = min(ends) if ends else len(problem_text)
        return problem_text[start:end].strip()

    lead_budget = max_chars // 3
    input_start = problem_text.find("Input")
    if input_start > 0:
        lead_start = max(0, input_start - lead_budget)
        paragraph_start = problem_text.rfind("\n\n", 0, input_start)
        if paragraph_start >= lead_start:
            previous_paragraph = problem_text.rfind("\n\n", 0, paragraph_start)
            if previous_paragraph >= 0:
                lead_start = previous_paragraph + 2
        lead = problem_text[lead_start:input_start].strip()
    else:
        lead = problem_text[:lead_budget].rstrip()

    parts = [lead] if lead else []
    for marker in markers:
        section = _section(marker)
        if not section:
            continue
        candidate = "\n\n".join(part for part in [*parts, section] if part)
        if len(candidate) <= max_chars:
            parts.append(section)
            continue
        remaining = max_chars - len("\n\n".join(parts)) - 2
        if remaining > 0:
            parts.append(section[:remaining].rstrip())
        break

    shrunk = "\n\n".join(part for part in parts if part).strip()
    return shrunk or problem_text[:max_chars]


def build_small_model_stdin_prompt(problem_text: str, *, compact: bool = False) -> str:
    if compact:
        return (
            "Solve the competitive-programming problem below.\n"
            "Use this compact skeleton.\n"
            "Output only Python code.\n"
            "Return a complete program.\n"
            "Never output only statements or fragments.\n"
            "Keep the code very short.\n"
            "Use short variable names.\n"
            "Do not write comments.\n"
            "Do not add blank lines.\n"
            "Append answer strings to out.\n"
            "Do not leave out empty.\n"
            "Start with `import sys` on the first line.\n\n"
            "import sys\n"
            "def solve():\n"
            "    d=sys.stdin.buffer.read().split()\n"
            "    out=[]\n"
            "    sys.stdout.write('\\n'.join(out))\n"
            "if __name__=='__main__': solve()\n\n"
            "Problem:\n"
            f"{problem_text}"
        )

    return (
        "Solve the competitive-programming problem below.\n"
        "Use this exact program skeleton and replace only the parsing and logic.\n"
        "Keep `solve()` and the final `sys.stdout.write` pattern.\n"
        "Output only Python code.\n\n"
        "Start with `import sys` on the first line.\n"
        "Do not explain the approach.\n"
        "Do not output `<think>`, markdown fences, comments outside the code, or prose.\n"
        "If you output any text before the code, the answer is wrong.\n\n"
        "import sys\n\n"
        "def solve():\n"
        "    # parse tokens from data\n"
        "    data = sys.stdin.read().strip().split()\n"
        "    # append each answer as a string into out\n"
        "    out = []\n"
        "    sys.stdout.write(\"\\n\".join(out))\n\n"
        "if __name__ == '__main__':\n"
        "    solve()\n\n"
        "Problem:\n"
        f"{problem_text}"
    )

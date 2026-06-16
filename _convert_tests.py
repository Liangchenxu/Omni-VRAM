"""
Automated script to convert unittest.TestCase → pytest style.
Converts all tests/*.py files in place.
"""
import re
import os
import glob

TESTS_DIR = os.path.join(os.path.dirname(__file__), "tests")

# Mapping of unittest assertion methods → pytest equivalents
ASSERTION_MAP = {
    # Simple comparisons
    r'self\.assertEqual\((.+?),\s*(.+?)\)': r'assert \1 == \2',
    r'self\.assertNotEqual\((.+?),\s*(.+?)\)': r'assert \1 != \2',
    r'self\.assertTrue\((.+?)\)': r'assert \1',
    r'self\.assertFalse\((.+?)\)': r'assert not \1',
    r'self\.assertIsNone\((.+?)\)': r'assert \1 is None',
    r'self\.assertIsNotNone\((.+?)\)': r'assert \1 is not None',
    r'self\.assertIn\((.+?),\s*(.+?)\)': r'assert \1 in \2',
    r'self\.assertNotIn\((.+?),\s*(.+?)\)': r'assert \1 not in \2',
    r'self\.assertIsInstance\((.+?),\s*(.+?)\)': r'assert isinstance(\1, \2)',
    r'self\.assertGreater\((.+?),\s*(.+?)\)': r'assert \1 > \2',
    r'self\.assertGreaterEqual\((.+?),\s*(.+?)\)': r'assert \1 >= \2',
    r'self\.assertLess\((.+?),\s*(.+?)\)': r'assert \1 < \2',
    r'self\.assertLessEqual\((.+?),\s*(.+?)\)': r'assert \1 <= \2',
    # Approximate
    r'self\.assertAlmostEqual\((.+?),\s*(.+?),\s*places=(\d+)\)': r'assert round(abs(\1 - \2), \3) == 0',
    r'self\.assertAlmostEqual\((.+?),\s*(.+?)\)': r'assert \1 == pytest.approx(\2)',
    # Raises
    r'self\.assertRaises\((\w+)\)': r'pytest.raises(\1)',
    # Regex
    r'self\.assertRegex\((.+?),\s*(.+?)\)': r'assert re.search(\2, \1)',
    # Length
    r'self\.assertLen\((.+?),\s*(.+?)\)': r'assert len(\1) == \2',
}


def convert_file(filepath):
    """Convert a single test file from unittest to pytest style."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # 1. Replace import unittest → import pytest (keep if already has pytest)
    has_pytest_import = 'import pytest' in content
    if 'import unittest' in content:
        content = content.replace('import unittest', 'import pytest')
    elif not has_pytest_import:
        # Add pytest import after other imports
        content = 'import pytest\n' + content

    # 2. Replace class inheritance
    content = re.sub(
        r'class (\w+)\(unittest\.TestCase\):',
        r'class \1:',
        content
    )

    # 3. Convert setUp → setup_method
    content = re.sub(r'def setUp\(self\):', 'def setup_method(self):', content)
    content = re.sub(r'def tearDown\(self\):', 'def teardown_method(self):', content)

    # 4. Convert assertions (order matters - do complex ones first)
    for pattern, replacement in ASSERTION_MAP.items():
        # Handle assertRaises context manager
        if 'assertRaises' in pattern:
            content = re.sub(
                r'with self\.assertRaises\((\w+)\)(.*?):',
                r'with pytest.raises(\1)\2:',
                content
            )
            content = re.sub(pattern, replacement, content)
        else:
            content = re.sub(pattern, replacement, content)

    # 5. Remove unittest.main() block
    content = re.sub(
        r"\nif __name__\s*==\s*['\"]__main__['\"]\s*:\s*\n\s*unittest\.main\(\)\s*\n?",
        '\n',
        content
    )

    # 6. Clean up any remaining unittest references in comments (leave them)
    # Only replace functional references
    content = re.sub(r'unittest\.TestCase', 'pytest', content)  # in comments

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def main():
    test_files = glob.glob(os.path.join(TESTS_DIR, "test_*.py")) + \
                 glob.glob(os.path.join(TESTS_DIR, "benchmark_*.py")) + \
                 glob.glob(os.path.join(TESTS_DIR, "quick_test_*.py"))
    
    converted = 0
    for filepath in sorted(test_files):
        filename = os.path.basename(filepath)
        if convert_file(filepath):
            print(f"  ✓ {filename}")
            converted += 1
        else:
            print(f"  - {filename} (no changes)")
    
    print(f"\nConverted {converted}/{len(test_files)} files")


if __name__ == "__main__":
    main()
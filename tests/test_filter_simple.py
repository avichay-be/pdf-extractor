#!/usr/bin/env python3
"""
Simple standalone test for query filtering (no dependencies).
"""


def filter_outlines_by_query(outline_info: list, query: str) -> list:
    """
    Filter outline sections by query string (case-insensitive partial match).

    Args:
        outline_info: List of outline metadata dicts
        query: Search query string

    Returns:
        Filtered list of outline metadata, or original list if no matches found
    """
    if not outline_info or not query:
        return outline_info

    query_lower = query.lower()
    filtered = [
        outline for outline in outline_info
        if query_lower in outline['title'].lower()
    ]

    # If no matches found, return all outlines (fallback)
    return filtered if filtered else outline_info


def test_filter_function():
    """Run manual tests on the filter function."""
    print("="*80)
    print("QUERY FILTER TESTS")
    print("="*80)

    # Sample outline metadata
    outline_info = [
        {
            'title': 'דוחות כספיים',
            'page': 0,
            'chunk_indices': [0, 1, 2]
        },
        {
            'title': 'דוח דירקטוריון',
            'page': 30,
            'chunk_indices': [3, 4]
        },
        {
            'title': 'תקציר',
            'page': 50,
            'chunk_indices': [5]
        },
        {
            'title': 'Financial Reports',
            'page': 60,
            'chunk_indices': [6, 7]
        }
    ]

    tests = [
        ("Filter by 'דוחות כספיים' (default)", "דוחות כספיים", 1, ['דוחות כספיים']),
        ("Filter by 'דוח דירקטוריון'", "דוח דירקטוריון", 1, ['דוח דירקטוריון']),
        ("Filter partial match 'דוח'", "דוח", 2, ['דוחות כספיים', 'דוח דירקטוריון']),
        ("Filter case-insensitive 'financial'", "financial", 1, ['Financial Reports']),
        ("No match returns all", "nonexistent", 4, None),
        ("Empty query returns all", "", 4, None),
        ("None query returns all", None, 4, None),
    ]

    passed = 0
    failed = 0

    for test_name, query, expected_count, expected_titles in tests:
        print(f"\n{test_name}")
        print(f"  Query: '{query}'")

        result = filter_outlines_by_query(outline_info, query)

        print(f"  Expected count: {expected_count}, Got: {len(result)}")

        if len(result) == expected_count:
            if expected_titles:
                actual_titles = [r['title'] for r in result]
                if actual_titles == expected_titles:
                    print("  ✅ PASS")
                    passed += 1
                else:
                    print(f"  ❌ FAIL - Expected titles: {expected_titles}, Got: {actual_titles}")
                    failed += 1
            else:
                print("  ✅ PASS")
                passed += 1
        else:
            print(f"  ❌ FAIL")
            failed += 1

    # Test with None outline_info
    print(f"\nTest with None outline_info")
    result = filter_outlines_by_query(None, "query")
    if result is None:
        print("  ✅ PASS")
        passed += 1
    else:
        print("  ❌ FAIL")
        failed += 1

    # Test with empty outline_info
    print(f"\nTest with empty outline_info")
    result = filter_outlines_by_query([], "query")
    if result == []:
        print("  ✅ PASS")
        passed += 1
    else:
        print("  ❌ FAIL")
        failed += 1

    print("\n" + "="*80)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("="*80)

    return failed == 0


if __name__ == "__main__":
    success = test_filter_function()
    exit(0 if success else 1)

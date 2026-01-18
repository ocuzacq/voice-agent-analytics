#!/usr/bin/env python3
"""
Minimal Test Harness for Vacatia Analytics Framework v3.1

Validates:
- Schema validation rules
- Metrics computation correctness
- NL extraction structure
- Pipeline integration

Uses only standard library (no pytest required).
"""

import argparse
import json
import sys
from pathlib import Path

# Add tools directory to path for imports
TOOLS_DIR = Path(__file__).parent
sys.path.insert(0, str(TOOLS_DIR))

# Import modules under test
from compute_metrics import load_analyses, generate_report, safe_rate, safe_stats
from extract_nl_fields import load_v3_analyses, extract_nl_summary


# Test configuration
TEST_FILE_PATTERN = "test-v3-*.json"
ANALYSES_DIR = TOOLS_DIR.parent / "analyses"
EXPECTED_METRICS_PATH = TOOLS_DIR / "test_data" / "expected_metrics.json"


class TestResult:
    """Simple test result tracker."""
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.message = ""
        self.details = ""

    def fail(self, message: str, details: str = ""):
        self.passed = False
        self.message = message
        self.details = details
        return self

    def success(self, message: str = ""):
        self.passed = True
        self.message = message
        return self


class TestHarness:
    """Test harness for analytics framework."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: list[TestResult] = []
        self.test_analyses: list[dict] = []
        self.expected: dict = {}

    def load_test_data(self) -> bool:
        """Load test v3 analyses and expected metrics."""
        # Load expected metrics
        if not EXPECTED_METRICS_PATH.exists():
            print(f"ERROR: Expected metrics not found: {EXPECTED_METRICS_PATH}")
            return False

        with open(EXPECTED_METRICS_PATH, 'r') as f:
            self.expected = json.load(f)

        # Load test v3 analyses (only test-v3-* files)
        self.test_analyses = []
        for f in ANALYSES_DIR.glob(TEST_FILE_PATTERN):
            with open(f, 'r') as fp:
                data = json.load(fp)
                if data.get("schema_version") == "v3":
                    self.test_analyses.append(data)

        if not self.test_analyses:
            print(f"ERROR: No test analyses found matching {TEST_FILE_PATTERN}")
            return False

        if self.verbose:
            print(f"Loaded {len(self.test_analyses)} test analyses")
            print(f"Expected metrics loaded from {EXPECTED_METRICS_PATH}")

        return True

    # =========================================================================
    # Category A: Schema Validation Tests
    # =========================================================================

    def test_outcome_failure_consistency(self) -> TestResult:
        """Test: resolved outcome should have failure_point=none, non-resolved should have specific failure_point."""
        result = TestResult("outcome_failure_consistency")
        valid_count = 0
        errors = []

        for a in self.test_analyses:
            call_id = a.get("call_id")
            outcome = a.get("outcome")
            failure_point = a.get("failure_point")

            if outcome == "resolved":
                if failure_point == "none" or failure_point is None:
                    valid_count += 1
                else:
                    errors.append(f"{call_id}: resolved but failure_point={failure_point}")
            else:
                if failure_point and failure_point != "none":
                    valid_count += 1
                else:
                    errors.append(f"{call_id}: {outcome} but failure_point={failure_point}")

        expected_valid = self.expected.get("schema_validation", {}).get("outcome_failure_consistency", {}).get("expected_valid", len(self.test_analyses))

        if valid_count == expected_valid and not errors:
            return result.success(f"{valid_count}/{len(self.test_analyses)} valid")
        else:
            return result.fail(f"{valid_count}/{expected_valid} valid", "\n".join(errors))

    def test_policy_gap_detail_population(self) -> TestResult:
        """Test: failure_point=policy_gap should have policy_gap_detail populated."""
        result = TestResult("policy_gap_detail_population")
        policy_gap_calls = [a for a in self.test_analyses if a.get("failure_point") == "policy_gap"]
        valid_count = 0
        errors = []

        for a in policy_gap_calls:
            call_id = a.get("call_id")
            detail = a.get("policy_gap_detail")
            if detail and isinstance(detail, dict):
                # Check required fields
                if detail.get("category") and detail.get("specific_gap"):
                    valid_count += 1
                else:
                    errors.append(f"{call_id}: policy_gap_detail missing required fields")
            else:
                errors.append(f"{call_id}: policy_gap_detail not populated")

        expected_valid = self.expected.get("schema_validation", {}).get("policy_gap_detail_population", {}).get("expected_valid", len(policy_gap_calls))

        if valid_count == expected_valid:
            return result.success(f"{valid_count}/{len(policy_gap_calls)} valid")
        else:
            return result.fail(f"{valid_count}/{expected_valid} valid", "\n".join(errors))

    def test_quality_score_bounds(self) -> TestResult:
        """Test: all quality scores should be in range 1-5."""
        result = TestResult("quality_score_bounds")
        score_fields = ["agent_effectiveness", "conversation_quality", "customer_effort"]
        errors = []

        for a in self.test_analyses:
            call_id = a.get("call_id")
            for field in score_fields:
                value = a.get(field)
                if value is not None:
                    if not isinstance(value, (int, float)) or value < 1 or value > 5:
                        errors.append(f"{call_id}: {field}={value} out of range")

        if not errors:
            return result.success("all in range")
        else:
            return result.fail(f"{len(errors)} violations", "\n".join(errors))

    def test_recoverable_miss_detail(self) -> TestResult:
        """Test: was_recoverable=true should have agent_miss_detail (soft rule for coaching)."""
        result = TestResult("recoverable_miss_detail")
        recoverable_calls = [a for a in self.test_analyses if a.get("was_recoverable") is True]
        valid_count = 0
        warnings = []

        for a in recoverable_calls:
            call_id = a.get("call_id")
            miss_detail = a.get("agent_miss_detail")
            if miss_detail:
                valid_count += 1
            else:
                warnings.append(f"{call_id}: recoverable but no agent_miss_detail")

        expected_valid = self.expected.get("schema_validation", {}).get("recoverable_miss_detail", {}).get("expected_valid", len(recoverable_calls))

        if valid_count == expected_valid:
            return result.success(f"{valid_count}/{len(recoverable_calls)} valid")
        else:
            return result.fail(f"{valid_count}/{expected_valid} valid", "\n".join(warnings))

    # =========================================================================
    # Category B: Metrics Computation Tests
    # =========================================================================

    def test_rate_calculations(self) -> TestResult:
        """Test: key rates match expected values."""
        result = TestResult("rate_calculations")
        report = generate_report(self.test_analyses)
        expected_rates = self.expected.get("key_rates", {})
        actual_rates = report.get("key_rates", {})
        errors = []

        for rate_name, expected_value in expected_rates.items():
            actual_value = actual_rates.get(rate_name)
            if actual_value is None:
                errors.append(f"{rate_name}: missing")
            elif abs(actual_value - expected_value) > 0.001:
                errors.append(f"{rate_name}: expected {expected_value}, got {actual_value}")

        if not errors:
            return result.success()
        else:
            return result.fail("rate mismatch", "\n".join(errors))

    def test_quality_statistics(self) -> TestResult:
        """Test: quality score statistics match expected values."""
        result = TestResult("quality_statistics")
        report = generate_report(self.test_analyses)
        expected_quality = self.expected.get("quality_scores", {})
        actual_quality = report.get("quality_scores", {})
        errors = []

        for score_name, expected_stats in expected_quality.items():
            actual_stats = actual_quality.get(score_name, {})

            # Check key statistics
            for stat in ["n", "mean", "median"]:
                expected_val = expected_stats.get(stat)
                actual_val = actual_stats.get(stat)
                if expected_val is not None:
                    if actual_val is None:
                        errors.append(f"{score_name}.{stat}: missing")
                    elif abs(actual_val - expected_val) > 0.01:
                        errors.append(f"{score_name}.{stat}: expected {expected_val}, got {actual_val}")

        if not errors:
            return result.success()
        else:
            return result.fail("statistics mismatch", "\n".join(errors))

    def test_outcome_distribution(self) -> TestResult:
        """Test: outcome distribution sums correctly and matches expected."""
        result = TestResult("outcome_distribution")
        report = generate_report(self.test_analyses)
        expected_outcomes = self.expected.get("outcome_distribution", {})
        actual_outcomes = report.get("outcome_distribution", {})
        errors = []

        # Check total sums to n
        total_count = sum(v.get("count", 0) for v in actual_outcomes.values())
        if total_count != len(self.test_analyses):
            errors.append(f"Total count {total_count} != {len(self.test_analyses)} analyses")

        # Check each outcome
        for outcome, expected_data in expected_outcomes.items():
            actual_data = actual_outcomes.get(outcome, {})
            if actual_data.get("count") != expected_data.get("count"):
                errors.append(f"{outcome}: expected count {expected_data.get('count')}, got {actual_data.get('count')}")
            if expected_data.get("rate") and abs(actual_data.get("rate", 0) - expected_data["rate"]) > 0.001:
                errors.append(f"{outcome}: expected rate {expected_data.get('rate')}, got {actual_data.get('rate')}")

        if not errors:
            return result.success()
        else:
            return result.fail("distribution mismatch", "\n".join(errors))

    def test_policy_gap_aggregation(self) -> TestResult:
        """Test: policy gap breakdown aggregates correctly."""
        result = TestResult("policy_gap_aggregation")
        report = generate_report(self.test_analyses)
        expected_pgb = self.expected.get("policy_gap_breakdown", {})
        actual_pgb = report.get("policy_gap_breakdown", {})
        errors = []

        # Check category counts
        expected_cats = expected_pgb.get("by_category", {})
        actual_cats = actual_pgb.get("by_category", {})

        for cat, expected_data in expected_cats.items():
            actual_data = actual_cats.get(cat, {})
            if actual_data.get("count") != expected_data.get("count"):
                errors.append(f"category {cat}: expected {expected_data.get('count')}, got {actual_data.get('count')}")

        # Check total
        actual_total = sum(d.get("count", 0) for d in actual_cats.values())
        expected_total = expected_pgb.get("total_gaps", 3)
        if actual_total != expected_total:
            errors.append(f"total gaps: expected {expected_total}, got {actual_total}")

        if not errors:
            return result.success()
        else:
            return result.fail("aggregation mismatch", "\n".join(errors))

    # =========================================================================
    # Category C: NL Extraction Tests
    # =========================================================================

    def test_by_failure_type_grouping(self) -> TestResult:
        """Test: NL extraction groups correctly by failure type."""
        result = TestResult("by_failure_type_grouping")
        nl_summary = extract_nl_summary(self.test_analyses)
        expected_nl = self.expected.get("nl_extraction", {})
        by_type = nl_summary.get("by_failure_type", {})
        errors = []

        expected_by_type = expected_nl.get("by_failure_type", {})
        for fp_type, expected_count in expected_by_type.items():
            actual_count = len(by_type.get(fp_type, []))
            if actual_count != expected_count:
                errors.append(f"{fp_type}: expected {expected_count}, got {actual_count}")

        if not errors:
            return result.success()
        else:
            return result.fail("grouping mismatch", "\n".join(errors))

    def test_verbatim_extraction(self) -> TestResult:
        """Test: all customer verbatims are captured."""
        result = TestResult("verbatim_extraction")
        nl_summary = extract_nl_summary(self.test_analyses)
        expected_count = self.expected.get("nl_extraction", {}).get("total_verbatims", 0)
        actual_count = len(nl_summary.get("all_verbatims", []))

        if actual_count == expected_count:
            return result.success(f"{actual_count} quotes")
        else:
            return result.fail(f"expected {expected_count}, got {actual_count}")

    def test_agent_miss_extraction(self) -> TestResult:
        """Test: all agent miss details are captured."""
        result = TestResult("agent_miss_extraction")
        nl_summary = extract_nl_summary(self.test_analyses)
        expected_count = self.expected.get("nl_extraction", {}).get("total_agent_misses", 0)
        actual_count = len(nl_summary.get("all_agent_misses", []))

        if actual_count == expected_count:
            return result.success(f"{actual_count} misses")
        else:
            return result.fail(f"expected {expected_count}, got {actual_count}")

    def test_policy_gap_details(self) -> TestResult:
        """Test: all policy gap details are captured."""
        result = TestResult("policy_gap_details")
        nl_summary = extract_nl_summary(self.test_analyses)
        expected_count = self.expected.get("nl_extraction", {}).get("total_policy_gap_details", 0)
        actual_count = len(nl_summary.get("policy_gap_details", []))

        if actual_count == expected_count:
            return result.success(f"{actual_count} gaps")
        else:
            return result.fail(f"expected {expected_count}, got {actual_count}")

    # =========================================================================
    # Category D: Integration Tests
    # =========================================================================

    def test_compute_metrics_output(self) -> TestResult:
        """Test: compute_metrics produces valid JSON output."""
        result = TestResult("compute_metrics_output")
        try:
            report = generate_report(self.test_analyses)
            # Verify JSON serializable
            json_str = json.dumps(report)
            # Verify has required sections
            required = ["metadata", "outcome_distribution", "key_rates", "quality_scores", "failure_analysis"]
            missing = [s for s in required if s not in report]
            if missing:
                return result.fail(f"missing sections: {missing}")
            return result.success("valid JSON")
        except Exception as e:
            return result.fail(str(e))

    def test_extract_nl_fields_output(self) -> TestResult:
        """Test: extract_nl_fields produces valid JSON output."""
        result = TestResult("extract_nl_fields_output")
        try:
            nl_summary = extract_nl_summary(self.test_analyses)
            # Verify JSON serializable
            json_str = json.dumps(nl_summary)
            # Verify has required sections
            required = ["metadata", "by_failure_type", "all_verbatims", "all_agent_misses", "policy_gap_details"]
            missing = [s for s in required if s not in nl_summary]
            if missing:
                return result.fail(f"missing sections: {missing}")
            return result.success("valid JSON")
        except Exception as e:
            return result.fail(str(e))

    def test_edge_case_empty_input(self) -> TestResult:
        """Test: handles empty input gracefully."""
        result = TestResult("edge_case_empty_input")
        try:
            report = generate_report([])
            if "error" in report:
                return result.success("handled gracefully")
            else:
                return result.fail("should return error for empty input")
        except Exception as e:
            return result.fail(f"crashed: {e}")

    def test_safe_rate_edge_cases(self) -> TestResult:
        """Test: safe_rate handles edge cases."""
        result = TestResult("safe_rate_edge_cases")
        errors = []

        # Division by zero
        if safe_rate(5, 0) is not None:
            errors.append("safe_rate(5, 0) should be None")

        # Normal case
        if safe_rate(1, 4) != 0.25:
            errors.append(f"safe_rate(1, 4) should be 0.25, got {safe_rate(1, 4)}")

        # Precision
        if safe_rate(1, 3, 2) != 0.33:
            errors.append(f"safe_rate(1, 3, 2) should be 0.33, got {safe_rate(1, 3, 2)}")

        if not errors:
            return result.success()
        else:
            return result.fail("edge case failures", "\n".join(errors))

    def test_safe_stats_edge_cases(self) -> TestResult:
        """Test: safe_stats handles edge cases."""
        result = TestResult("safe_stats_edge_cases")
        errors = []

        # Empty input
        stats = safe_stats([])
        if stats.get("n") != 0 or stats.get("mean") is not None:
            errors.append(f"empty input should have n=0, mean=None: {stats}")

        # Single value
        stats = safe_stats([5])
        if stats.get("n") != 1 or stats.get("mean") != 5 or stats.get("std") != 0.0:
            errors.append(f"single value stats incorrect: {stats}")

        # With None values (should be filtered)
        stats = safe_stats([1, None, 3, None, 5])
        if stats.get("n") != 3:
            errors.append(f"should filter None values: {stats}")

        if not errors:
            return result.success()
        else:
            return result.fail("edge case failures", "\n".join(errors))

    # =========================================================================
    # Test Runner
    # =========================================================================

    def run_category(self, category: str) -> list[TestResult]:
        """Run tests for a specific category."""
        tests_by_category = {
            "schema": [
                self.test_outcome_failure_consistency,
                self.test_policy_gap_detail_population,
                self.test_quality_score_bounds,
                self.test_recoverable_miss_detail,
            ],
            "metrics": [
                self.test_rate_calculations,
                self.test_quality_statistics,
                self.test_outcome_distribution,
                self.test_policy_gap_aggregation,
            ],
            "extraction": [
                self.test_by_failure_type_grouping,
                self.test_verbatim_extraction,
                self.test_agent_miss_extraction,
                self.test_policy_gap_details,
            ],
            "integration": [
                self.test_compute_metrics_output,
                self.test_extract_nl_fields_output,
                self.test_edge_case_empty_input,
                self.test_safe_rate_edge_cases,
                self.test_safe_stats_edge_cases,
            ],
        }

        tests = tests_by_category.get(category, [])
        results = []
        for test_fn in tests:
            results.append(test_fn())
        return results

    def run_all(self) -> list[TestResult]:
        """Run all test categories."""
        all_results = []
        for category in ["schema", "metrics", "extraction", "integration"]:
            all_results.extend(self.run_category(category))
        return all_results


def print_results(results: list[TestResult], category: str, verbose: bool = False):
    """Print test results for a category."""
    category_names = {
        "schema": "Schema Validation Tests",
        "metrics": "Metrics Computation Tests",
        "extraction": "NL Extraction Tests",
        "integration": "Integration Tests",
    }

    print(f"\nRunning: {category_names.get(category, category)}")
    for r in results:
        status = "\u2713" if r.passed else "\u2717"
        if r.passed:
            msg = f": PASS" + (f" ({r.message})" if r.message else "")
        else:
            msg = f": FAIL - {r.message}"
        print(f"  {status} {r.name}{msg}")
        if verbose and r.details:
            for line in r.details.split("\n"):
                print(f"      {line}")


def main():
    parser = argparse.ArgumentParser(
        description="Test harness for Vacatia Analytics Framework v3.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all tests
    python3 tools/test_framework.py

    # Run specific category
    python3 tools/test_framework.py --category schema
    python3 tools/test_framework.py --category metrics

    # Verbose output
    python3 tools/test_framework.py -v
        """
    )
    parser.add_argument("-c", "--category", choices=["schema", "metrics", "extraction", "integration"],
                        help="Run only tests in specified category")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed output for failures")

    args = parser.parse_args()

    # Banner
    print("=" * 72)
    print("VACATIA ANALYTICS FRAMEWORK - TEST HARNESS v3.1")
    print("=" * 72)

    # Initialize and load test data
    harness = TestHarness(verbose=args.verbose)
    if not harness.load_test_data():
        return 1

    print(f"\nTest Data: {len(harness.test_analyses)} v3 analyses loaded")

    # Run tests
    all_results = []
    if args.category:
        results = harness.run_category(args.category)
        print_results(results, args.category, args.verbose)
        all_results = results
    else:
        for category in ["schema", "metrics", "extraction", "integration"]:
            results = harness.run_category(category)
            print_results(results, category, args.verbose)
            all_results.extend(results)

    # Summary
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)

    print("\n" + "=" * 72)
    if passed == total:
        print(f"RESULTS: {passed}/{total} tests passed")
    else:
        print(f"RESULTS: {passed}/{total} tests passed ({total - passed} FAILED)")
    print("=" * 72)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

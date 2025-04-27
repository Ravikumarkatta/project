"""Script to run theological validation tests."""
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.theology.validator import TheologicalValidator
from src.utils.logger import setup_logger

logger = setup_logger("theology_validation", "logs/theology_validation.log")


def run_validation_tests(test_mode: bool = False):
    """Run theological validation tests.

    Args:
        test_mode: If True, run in test mode with predefined statements
    """
    try:
        validator = TheologicalValidator()

        if test_mode:
            # Test orthodox statements
            orthodox_statements = [
                "God exists in three persons: Father, Son, and Holy Spirit.",
                "Jesus Christ is fully God and fully man.",
                "Salvation is by grace through faith alone.",
                "Scripture is the inspired Word of God.",
            ]

            # Test heterodox statements
            heterodox_statements = [
                "You must earn your salvation through good works.",
                "Jesus was just a good teacher, not God.",
                "The Bible contains errors and myths.",
                "There are many paths to God.",
            ]

            # Run validation tests
            logger.info("Testing orthodox statements...")
            orthodox_results = validator.validate_batch(orthodox_statements)
            all_orthodox_valid = all(score >= 0.7 for score in orthodox_results)

            logger.info("Testing heterodox statements...")
            heterodox_results = validator.validate_batch(heterodox_statements)
            all_heterodox_invalid = all(score < 0.7 for score in heterodox_results)

            if all_orthodox_valid and all_heterodox_invalid:
                logger.info("All theological validation tests passed")
                return True
            else:
                logger.error("Some theological validation tests failed")
                if not all_orthodox_valid:
                    logger.error("Failed to validate orthodox statements")
                if not all_heterodox_invalid:
                    logger.error("Failed to reject heterodox statements")
                return False

        else:
            # Run integration tests with rules
            rules_dir = project_root / "src/theology/rules"
            if not rules_dir.exists():
                logger.error("Theological rules directory not found")
                return False

            rule_files = list(rules_dir.glob("*.json"))
            if not rule_files:
                logger.error("No theological rule files found")
                return False

            # Test each rule file
            for rule_file in rule_files:
                with open(rule_file) as f:
                    rules = json.load(f)

                # Test key statements
                statements = rules.get("key_statements", [])
                if statements:
                    results = validator.validate_batch(statements)
                    if not all(score >= 0.7 for score in results):
                        logger.error(f"Validation failed for rules in {rule_file.name}")
                        return False

            logger.info("All theological rule validations passed")
            return True

    except Exception as e:
        logger.error(f"Theological validation testing failed: {str(e)}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run theological validation tests")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with predefined statements",
    )
    parser.add_argument(
        "--integration-test",
        action="store_true",
        help="Run integration tests with rule files",
    )

    args = parser.parse_args()

    success = run_validation_tests(test_mode=args.test_mode)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

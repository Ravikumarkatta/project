#!/usr/bin/env python3
"""Script to validate theological consistency and run theological tests."""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.theology.validator import TheologicalValidator
from src.utils.logger import setup_logger

logger = setup_logger("theology_validation", "logs/theology_validation.log")

def validate_theological_rules():
    """Validate all theological rules for consistency."""
    validator = TheologicalValidator()
    rules_dir = os.path.join(project_root, "src", "theology", "rules")
    
    if not os.path.exists(rules_dir):
        logger.error("Theological rules directory not found")
        return False
    
    # Load and validate each rule file
    rule_files = [f for f in os.listdir(rules_dir) if f.endswith('.json')]
    
    if not rule_files:
        logger.error("No theological rule files found")
        return False
    
    all_valid = True
    for rule_file in rule_files:
        try:
            with open(os.path.join(rules_dir, rule_file), 'r') as f:
                rule_data = json.load(f)
            
            # Validate rule structure
            required_fields = ["key_statements", "keywords", "references"]
            missing_fields = [field for field in required_fields if field not in rule_data]
            
            if missing_fields:
                logger.error(f"Rule file {rule_file} missing required fields: {missing_fields}")
                all_valid = False
                continue
            
            # Validate content
            if not rule_data["key_statements"]:
                logger.error(f"Rule file {rule_file} has no key statements")
                all_valid = False
            
            if not rule_data["keywords"]:
                logger.error(f"Rule file {rule_file} has no keywords")
                all_valid = False
            
            if not rule_data["references"]:
                logger.error(f"Rule file {rule_file} has no references")
                all_valid = False
            
            # Test rule application
            test_text = " ".join(rule_data["key_statements"])
            validation_result = validator.validate_text(test_text)
            
            if not validation_result["valid"]:
                logger.error(
                    f"Rule {rule_file} failed self-validation with errors: "
                    f"{validation_result['errors']}"
                )
                all_valid = False
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in rule file: {rule_file}")
            all_valid = False
        except Exception as e:
            logger.error(f"Error processing rule file {rule_file}: {e}")
            all_valid = False
    
    return all_valid

def run_integration_tests():
    """Run integration tests for the theological system."""
    validator = TheologicalValidator()
    test_cases_path = os.path.join(project_root, "tests", "data", "theological_test_cases.json")
    
    try:
        with open(test_cases_path, 'r') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        logger.error(f"Test cases file not found: {test_cases_path}")
        return False
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in test cases file: {test_cases_path}")
        return False
    
    all_passed = True
    for case in test_cases:
        try:
            result = validator.validate_text(case["input"])
            expected_valid = case["expected_valid"]
            
            if result["valid"] != expected_valid:
                logger.error(
                    f"Test case failed: {case['name']}\n"
                    f"Expected valid: {expected_valid}, got: {result['valid']}\n"
                    f"Input: {case['input'][:100]}...\n"
                    f"Errors: {result.get('errors', [])}"
                )
                all_passed = False
        except Exception as e:
            logger.error(f"Error running test case {case.get('name', 'unknown')}: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Main entry point for theology validation."""
    parser = argparse.ArgumentParser(description="Validate theological rules and run tests")
    parser.add_argument(
        "--test-mode", 
        action="store_true",
        help="Run in test mode (validate rules only)"
    )
    parser.add_argument(
        "--integration-test",
        action="store_true",
        help="Run integration tests"
    )
    
    args = parser.parse_args()
    
    # Always validate rules
    logger.info("Validating theological rules...")
    rules_valid = validate_theological_rules()
    
    if args.integration_test:
        logger.info("Running integration tests...")
        tests_passed = run_integration_tests()
        success = rules_valid and tests_passed
    else:
        success = rules_valid
    
    if success:
        logger.info("All theological validations passed successfully")
    else:
        logger.error("Theological validation failed")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
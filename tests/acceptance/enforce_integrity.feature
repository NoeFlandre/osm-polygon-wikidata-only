Feature: Enforcing link integrity
  Executable specification for issue #115.

  # Regression for #88.
  Scenario: A dry run reports violations without rewriting files
    Given a processed tree with a polygon article link to an unknown QID
    When I run enforce-integrity with --dry-run
    Then the summary reports one rejected polygon article link
    And the processed files are byte-identical and no audit is written

  # Regression for #88.
  Scenario: A real run rewrites the violations and writes an audit
    Given a processed tree with a polygon article link to an unknown QID
    When I run enforce-integrity
    Then the summary reports one rejected polygon article link
    And an audit file is written

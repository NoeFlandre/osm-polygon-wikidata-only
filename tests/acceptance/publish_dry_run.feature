Feature: Publishing statistics against a fake Hub
  Executable specification for issue #115: a fake in-memory Hub stands in
  for Hugging Face, so nothing is ever uploaded.

  Scenario: A dry run plans the release and uploads nothing
    Given a staged V2 release and a fake Hub with a prior card snapshot
    When I publish the statistics with dry-run
    Then the planned operations include the V2 card, report, and map assets
    And nothing is uploaded or committed to the Hub

  Scenario: Republishing an unchanged release against its own snapshot is a no-op
    Given a staged V2 release and a fake Hub with a prior card snapshot
    And the release was already published to the fake Hub
    When I publish the statistics again
    Then no file is reported as changed and no second commit is made

  # Regression for #15 and #21.
  Scenario: A commit URL revision is normalised before verification
    Given released files already present on a fake Hub
    When I verify the release with a full commit URL revision
    Then every Hub file API receives the bare commit hash

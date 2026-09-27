Feature: Dataset cards agree with the release statistics
  Executable specification for issue #115.

  # Regression for #40.
  Scenario Outline: The <version> card continent table and assets match the stats
    Given release statistics fixtures for the <version> dataset
    When I render the <version> dataset card
    Then the card continent table lists every continent from the stats
    And the continent text coverage sums to the headline figure
    And the card references every released map asset

    Examples:
      | version |
      | V1      |
      | V2      |

  # Regression for #40.
  Scenario: A card whose continent table contradicts its headline is refused
    Given release statistics fixtures for the V1 dataset
    When the rendered card headline disagrees with its continent table
    Then the release consistency guard refuses the card

"""Search decision-making: coverage planning and the survivor belief map.

The planner decides *where to fly*; the belief map decides *where it is worth
flying again*.  They are separate because a lawnmower plan is complete and
predictable, and a belief-driven plan is neither - the useful system runs the
lawnmower over the belief-weighted ordering, so completeness is preserved and
endurance is spent where people are most likely to be.
"""

from sar.decision.coverage import (BeliefMap, BoustrophedonPlanner, CoverageGrid,
                                   Lane, PriorSource, SurveyPlan)

__all__ = ["BeliefMap", "BoustrophedonPlanner", "CoverageGrid", "Lane",
           "PriorSource", "SurveyPlan"]

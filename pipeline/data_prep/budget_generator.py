"""Generate budget trap negative examples."""
from typing import List, Dict, Any

from .base import BaseProcessor
from .config import DataPrepConfig
from .models import BudgetTrap


class BudgetTrapGenerator(BaseProcessor):
    """Generates budget conflict negative examples.

    Budget traps are query-property pairs where:
    - Budget constraint is explicitly stated in query
    - Property has attractive amenities
    - BUT: Rent exceeds budget (typical user error scenario)

    These teach the model to strictly enforce budget constraints.
    """

    def __init__(self, config: DataPrepConfig):
        super().__init__(config)
        self.budgets = [4000, 5000, 6000, 7000]
        self.amenities = [
            "獨立洗衣機", "獨洗", "陽台", "電梯",
            "台電計費", "新裝潢", "近興大"
        ]

    def run(self, target_count: int = 500) -> List[BudgetTrap]:
        """Generate budget trap examples.

        Args:
            target_count: Approximate number of traps to generate

        Returns:
            List of BudgetTrap objects
        """
        self.log_step(f"Generating ~{target_count} budget trap examples")

        traps = []

        # Strategy 1: Perfect amenities but over budget
        self.log_step("Strategy 1: Over-budget with amenities")
        for budget in self.budgets:
            for amenity in self.amenities:
                trap = BudgetTrap(
                    query=f"想找{budget}以內，要有{amenity}的套房",
                    property_id=f"trap_over_budget_{budget}_{amenity}",
                    user_budget=float(budget),
                    property_rent=float(budget + 5000),
                )
                traps.append(trap)

        # Strategy 2: Perfect budget but missing critical amenity
        self.log_step("Strategy 2: Missing required amenity")
        for budget in self.budgets:
            for amenity in self.amenities:
                trap = BudgetTrap(
                    query=f"預算{budget}，一定要有{amenity}",
                    property_id=f"trap_missing_amenity_{budget}_{amenity}",
                    user_budget=float(budget),
                    property_rent=float(budget - 500),
                )
                traps.append(trap)

        # Limit to target
        traps = traps[:target_count]
        self.log_result("Total traps generated", len(traps))

        # Save
        output_path = self.config.checkpoint_dir / "budget_traps.json"
        self._save_traps(traps, output_path)

        return traps

    @staticmethod
    def _save_traps(traps: List[BudgetTrap], path) -> None:
        """Save budget traps to JSON."""
        import json
        data = [
            {
                "query": t.query,
                "property_id": t.property_id,
                "user_budget": t.user_budget,
                "property_rent": t.property_rent,
                "type": "budget_trap",
            }
            for t in traps
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

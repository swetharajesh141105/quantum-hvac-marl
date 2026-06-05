"""
Quantum-Inspired QUBO Optimizer
=================================
Formulates the 24-hour AC scheduling problem as a QUBO
(Quadratic Unconstrained Binary Optimization) problem and
solves it using quantum-inspired techniques.

CONCEPT EXPLANATION (for interviews):
--------------------------------------
QUBO = Quadratic Unconstrained Binary Optimization

The problem: decide for each of 24 hours whether AC should be
ON (1) or OFF (0). That's 2^24 = 16 million possible schedules.

QUBO formulates this as minimizing:
    E(x) = Σᵢ aᵢxᵢ + Σᵢ<ⱼ bᵢⱼ xᵢxⱼ

Where:
  xᵢ ∈ {0, 1}  — AC on/off for hour i
  aᵢ           — cost of running AC in hour i (energy price, carbon intensity)
  bᵢⱼ          — interaction between hours i,j (e.g., thermal inertia means
                  hour 2 is cheaper if hour 1 was already cooled)

This is exactly the form that quantum annealers (D-Wave) and
quantum-approximate optimization algorithms (QAOA) can solve.

We use a CLASSICAL SIMULATED ANNEALING solver (via Qiskit's
optimization module) as a quantum-inspired approach.

WHY THIS MATTERS:
As buildings get more complex (1000+ rooms, renewable sources,
tariff structures), classical solvers take hours. Quantum solvers
can find near-optimal solutions in milliseconds.
"""

import numpy as np
from typing import List, Dict, Tuple
import json


class QUBOScheduler:
    """
    Quantum-inspired HVAC scheduling optimizer.
    
    Input: predicted occupancy for next 24 hours + energy prices
    Output: optimal ON/OFF schedule for each hour
    
    Method: Simulated Annealing (quantum-inspired, no real QC needed)
    """

    def __init__(self, n_hours: int = 24):
        self.n_hours = n_hours

    def build_qubo_matrix(
        self,
        occupancy_forecast: List[float],
        energy_prices: List[float],
        carbon_intensity: List[float],
        outdoor_temp_forecast: List[float],
    ) -> np.ndarray:
        """
        Build the QUBO Q-matrix for the scheduling problem.
        
        Q matrix is n×n where Q[i][j] represents the cost/benefit
        of having AC ON in both hour i and hour j simultaneously.
        
        Diagonal Q[i][i] = linear cost of AC being ON in hour i
        Off-diagonal Q[i][j] = interaction term between hours i and j
        
        We minimize: x^T Q x
        """
        Q = np.zeros((self.n_hours, self.n_hours))

        for i in range(self.n_hours):
            # ---- Diagonal terms (linear costs per hour) ----

            # Energy cost: higher price → more penalty for being ON
            energy_penalty = energy_prices[i] * 2.0

            # Carbon penalty: dirty grid → more penalty
            carbon_penalty = carbon_intensity[i] * 1.5

            # Occupancy benefit: reward for being ON when occupied
            # (negative cost = benefit)
            occupancy_benefit = -occupancy_forecast[i] * 3.0

            # Temperature pressure: if it will be hot, there's pressure to cool
            temp_pressure = max(0, outdoor_temp_forecast[i] - 28) * 0.5

            Q[i][i] = energy_penalty + carbon_penalty + occupancy_benefit + temp_pressure

        for i in range(self.n_hours):
            for j in range(i + 1, self.n_hours):
                # ---- Off-diagonal terms (interactions between hours) ----

                # Thermal inertia: if AC was ON last hour, current hour
                # needs less cooling (room is already cold)
                # Only adjacent hours have strong thermal coupling
                if abs(i - j) == 1:
                    thermal_benefit = -0.5  # running back-to-back is efficient
                    Q[i][j] = thermal_benefit
                    Q[j][i] = thermal_benefit

                # Penalty for running AC during consecutive unoccupied hours
                if occupancy_forecast[i] < 0.1 and occupancy_forecast[j] < 0.1:
                    if abs(i - j) <= 2:
                        Q[i][j] += 1.0
                        Q[j][i] += 1.0

        return Q

    def simulated_annealing_solve(
        self, Q: np.ndarray,
        n_iterations: int = 10000,
        T_start: float = 10.0,
        T_end: float = 0.01
    ) -> Tuple[np.ndarray, float]:
        """
        Simulated Annealing — the quantum-inspired solver.
        
        HOW IT WORKS:
        1. Start with random schedule (x)
        2. Compute energy E = x^T Q x
        3. Flip one bit randomly → new schedule x'
        4. If E(x') < E(x): always accept (better solution)
        5. If E(x') > E(x): accept with probability exp(-ΔE/T)
           T = temperature, starts high (accept bad moves → explore)
           T decreases over time → fewer bad moves → exploit
        
        This mimics quantum tunneling: the algorithm can "tunnel"
        through energy barriers to find the global minimum.
        
        ANALOGY: Like annealing metal — start hot (atoms move freely),
        cool slowly (atoms settle into lowest energy crystal structure).
        """
        n = len(Q)
        # Random initial solution
        x = np.random.randint(0, 2, n).astype(float)
        best_x = x.copy()
        best_energy = self._compute_energy(x, Q)
        current_energy = best_energy

        cooling_rate = (T_end / T_start) ** (1 / n_iterations)
        T = T_start

        for iteration in range(n_iterations):
            # Randomly flip one bit
            idx = np.random.randint(n)
            x_new = x.copy()
            x_new[idx] = 1 - x_new[idx]

            new_energy = self._compute_energy(x_new, Q)
            delta_E = new_energy - current_energy

            # Accept or reject
            if delta_E < 0:
                x = x_new
                current_energy = new_energy
            else:
                # Metropolis criterion — accept bad moves probabilistically
                accept_prob = np.exp(-delta_E / max(T, 1e-10))
                if np.random.random() < accept_prob:
                    x = x_new
                    current_energy = new_energy

            # Track best solution found
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()

            T *= cooling_rate

        return best_x, best_energy

    def _compute_energy(self, x: np.ndarray, Q: np.ndarray) -> float:
        """QUBO energy: E = x^T Q x"""
        return float(x.T @ Q @ x)

    def optimize_schedule(
        self,
        occupancy_forecast: List[float] = None,
        energy_prices: List[float] = None,
        carbon_intensity: List[float] = None,
        outdoor_temp_forecast: List[float] = None,
    ) -> Dict:
        """
        Main entry point: given forecasts, return optimized schedule.
        
        Returns:
            {
                "schedule": [0, 1, 1, 0, ...],  # 24 binary values
                "on_hours": [9, 10, 11, ...],
                "total_cost": float,
                "energy_savings_vs_always_on": float (%)
            }
        """
        # Default realistic values if not provided
        if occupancy_forecast is None:
            occupancy_forecast = self._default_occupancy()
        if energy_prices is None:
            energy_prices = self._default_energy_prices()
        if carbon_intensity is None:
            carbon_intensity = self._default_carbon_intensity()
        if outdoor_temp_forecast is None:
            outdoor_temp_forecast = self._default_outdoor_temps()

        print("Building QUBO matrix...")
        Q = self.build_qubo_matrix(
            occupancy_forecast, energy_prices,
            carbon_intensity, outdoor_temp_forecast
        )

        print("Running Simulated Annealing (quantum-inspired solver)...")
        schedule, energy = self.simulated_annealing_solve(Q, n_iterations=5000)

        # Post-process: force AC on when high occupancy
        for i in range(self.n_hours):
            if occupancy_forecast[i] > 0.7:
                schedule[i] = 1  # must be on when very occupied

        schedule = schedule.astype(int).tolist()
        on_hours = [h for h, s in enumerate(schedule) if s == 1]

        # Calculate savings vs always-on baseline
        always_on_cost = sum(energy_prices)
        our_cost = sum(energy_prices[h] for h in on_hours)
        savings_pct = (always_on_cost - our_cost) / always_on_cost * 100

        result = {
            "schedule": schedule,
            "on_hours": on_hours,
            "off_hours": [h for h, s in enumerate(schedule) if s == 0],
            "total_on_hours": len(on_hours),
            "qubo_energy": float(energy),
            "estimated_savings_vs_always_on_pct": round(savings_pct, 1),
        }

        return result

    # ------------------------------------------------------------------ #
    #  Default forecasts (can be replaced with LSTM predictions)          #
    # ------------------------------------------------------------------ #

    def _default_occupancy(self) -> List[float]:
        """Typical office occupancy pattern (0=empty, 1=full)."""
        pattern = [0] * 8 + [0.3, 0.7, 0.9, 0.9, 0.9, 0.5, 0.9, 0.9,
                              0.9, 0.8, 0.5, 0.3] + [0.1] * 4
        return pattern[:self.n_hours]

    def _default_energy_prices(self) -> List[float]:
        """Time-of-use tariff (₹/kWh). Peak: 18:00-22:00."""
        prices = []
        for h in range(self.n_hours):
            if 18 <= h <= 22:
                prices.append(8.0)  # peak rate
            elif 6 <= h < 18:
                prices.append(5.5)  # normal rate
            else:
                prices.append(3.0)  # off-peak rate
        return prices

    def _default_carbon_intensity(self) -> List[float]:
        """Grid carbon intensity (kg CO2/kWh). Lower during solar hours."""
        intensities = []
        for h in range(self.n_hours):
            if 10 <= h <= 16:
                intensities.append(0.3)  # solar peak
            elif 6 <= h < 10 or 16 < h < 20:
                intensities.append(0.5)  # mixed
            else:
                intensities.append(0.7)  # coal/gas heavy
        return intensities

    def _default_outdoor_temps(self) -> List[float]:
        """Daily temperature cycle (Chennai, India)."""
        return [28 + 5 * np.sin((h - 14) * np.pi / 12) for h in range(self.n_hours)]


def print_schedule(result: Dict):
    """Pretty-print the optimized schedule."""
    print("\n" + "=" * 60)
    print("QUBO OPTIMIZED AC SCHEDULE")
    print("=" * 60)

    schedule = result["schedule"]
    for h in range(24):
        bar = "█" * 20 if schedule[h] else "░" * 20
        status = "ON " if schedule[h] else "OFF"
        print(f"  {h:02d}:00  [{status}]  {bar}")

    print(f"\n  Total ON hours: {result['total_on_hours']}/24")
    print(f"  ON hours: {result['on_hours']}")
    print(f"  Estimated savings vs always-on: {result['estimated_savings_vs_always_on_pct']}%")
    print(f"  QUBO objective value: {result['qubo_energy']:.4f}")


if __name__ == "__main__":
    print("Quantum-Inspired QUBO Optimizer")
    print("Using: Simulated Annealing (quantum-inspired, no real QC needed)\n")

    scheduler = QUBOScheduler(n_hours=24)
    result = scheduler.optimize_schedule()
    print_schedule(result)

    # Save schedule as JSON (will be fed to MARL agents)
    with open("optimized_schedule.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSchedule saved to optimized_schedule.json")
    print("✅ QUBO optimization complete!")

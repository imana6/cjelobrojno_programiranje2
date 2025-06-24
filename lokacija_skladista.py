"""
FACILITY LOCATION PROBLEM (Uncapacitated)

Problem: Kompanija želi otvoriti distribucijske centre u različitim gradovima kako bi 
opskrbila svoje kupce uz minimalne ukupne troškove. Svaki centar ima fiksni trošak
otvaranja i varijabilni trošak transporta do kupaca.

Varijable:
- yi ∈ {0,1}: 1 ako otvaramo facility i, 0 inače  
- xij ∈ [0,1]: frakcija potražnje kupca j koju zadovoljava facility i

Cilj: Minimizirati ukupne troškove (fiksni + transport)

Ograničenja:
- Svaki kupac mora biti opskrbljen
- Možemo opskrbiti kupca samo iz otvorenog centra
- Binarne varijable za facility
"""

import numpy as np
import time
from scipy.optimize import linprog
from typing import List, Tuple, Optional
import heapq
import math

class FacilityLocationProblem:
    def __init__(self, n_facilities: int, n_customers: int, seed: int = 42):
        """
        n_facilities: broj mogućih lokacija za centre
        n_customers: broj kupaca
        """
        np.random.seed(seed)
        self.n_facilities = n_facilities
        self.n_customers = n_customers
        
        # Fiksni troškovi otvaranja facility-ja (200-1000)
        self.fixed_costs = np.random.uniform(200, 1000, n_facilities)
        
        # Transport troškovi (10-100 po jedinici udaljenosti)
        # Simuliramo koordinate i udaljenosti
        facility_coords = np.random.uniform(0, 100, (n_facilities, 2))
        customer_coords = np.random.uniform(0, 100, (n_customers, 2))
        
        self.transport_costs = np.zeros((n_facilities, n_customers))
        for i in range(n_facilities):
            for j in range(n_customers):
                distance = np.linalg.norm(facility_coords[i] - customer_coords[j])
                self.transport_costs[i, j] = distance * np.random.uniform(0.5, 2.0)
        
        # Potražnja kupaca
        self.demands = np.random.uniform(10, 50, n_customers)
        
        print(f"=== FACILITY LOCATION PROBLEM ===")
        print(f"Facilities: {n_facilities}, Customers: {n_customers}")
        print(f"Fiksni troškovi: {self.fixed_costs.min():.0f} - {self.fixed_costs.max():.0f}")
        print(f"Transport troškovi: {self.transport_costs.min():.2f} - {self.transport_costs.max():.2f}")

class BranchAndBoundSolver:
    def __init__(self, problem: FacilityLocationProblem):
        self.problem = problem
        self.best_solution = None
        self.best_value = float('inf')
        self.nodes_explored = 0
        self.max_nodes = 1000  # Ograničimo broj čvorova
        
    def solve_lp_relaxation(self, fixed_facilities: List[int], forbidden_facilities: List[int]) -> Tuple[float, np.ndarray]:
        """Rješava LP relaksaciju s fiksiranim facility varijablama"""
        n_vars = self.problem.n_facilities + self.problem.n_facilities * self.problem.n_customers
        
        # Ciljna funkcija: sum(fi*yi) + sum(cij*xij)
        c = np.concatenate([
            self.problem.fixed_costs,  # yi varijable
            self.problem.transport_costs.flatten()  # xij varijable
        ])
        
        # Ograničenja: sum_i(xij) = 1 za svaki j (svaki kupac mora biti opskrbljen)
        A_eq = []
        b_eq = []
        
        for j in range(self.problem.n_customers):
            constraint = np.zeros(n_vars)
            for i in range(self.problem.n_facilities):
                xij_index = self.problem.n_facilities + i * self.problem.n_customers + j
                constraint[xij_index] = 1
            A_eq.append(constraint)
            b_eq.append(1)
        
        # Ograničenja: xij <= yi za sve i,j
        A_ub = []
        b_ub = []
        
        for i in range(self.problem.n_facilities):
            for j in range(self.problem.n_customers):
                constraint = np.zeros(n_vars)
                constraint[i] = -1  # -yi
                xij_index = self.problem.n_facilities + i * self.problem.n_customers + j
                constraint[xij_index] = 1  # +xij
                A_ub.append(constraint)
                b_ub.append(0)
        
        # Ograničenja varijabli
        bounds = []
        
        # yi varijable
        for i in range(self.problem.n_facilities):
            if i in fixed_facilities:
                bounds.append((1, 1))
            elif i in forbidden_facilities:
                bounds.append((0, 0))
            else:
                bounds.append((0, 1))
        
        # xij varijable
        for _ in range(self.problem.n_facilities * self.problem.n_customers):
            bounds.append((0, 1))
        
        A_eq = np.array(A_eq) if A_eq else None
        b_eq = np.array(b_eq) if b_eq else None
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=bounds, method='highs')
        
        if result.success:
            return result.fun, result.x
        else:
            return float('inf'), None
    
    def is_integer_solution(self, x: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Provjerava je li rješenje cjelobrojno za yi varijable"""
        if x is None:
            return False
        y_vars = x[:self.problem.n_facilities]
        return all(abs(yi - round(yi)) < tolerance for yi in y_vars)
    
    def branch_and_bound(self) -> Tuple[Optional[np.ndarray], float, int]:
        """Glavna B&B funkcija"""
        # Priority queue: (lower_bound, node_id, fixed_facilities, forbidden_facilities)
        queue = [(0, 0, [], [])]
        node_counter = 0
        
        while queue and self.nodes_explored < self.max_nodes:
            lower_bound, node_id, fixed_facilities, forbidden_facilities = heapq.heappop(queue)
            self.nodes_explored += 1
            
            if lower_bound >= self.best_value:
                continue  # Pruning
            
            # Riješi LP relaksaciju
            obj_value, solution = self.solve_lp_relaxation(fixed_facilities, forbidden_facilities)
            
            if obj_value >= self.best_value:
                continue  
            
            if self.is_integer_solution(solution):
                # Našli smo novi najbolji cjelobrojni rezultat
                self.best_value = obj_value
                self.best_solution = solution
                continue
            
            # Branching - pronađi najfrakcionalniju yi varijablu
            y_vars = solution[:self.problem.n_facilities]
            fractional_values = [(abs(yi - round(yi)), i) for i, yi in enumerate(y_vars)]
            fractional_values.sort(reverse=True)
            
            if fractional_values[0][0] > 1e-6:  # Ima smisla granati
                branch_var = fractional_values[0][1]
                
                # Stvori dva nova čvora
                node_counter += 1
                new_fixed = fixed_facilities + [branch_var]
                heapq.heappush(queue, (obj_value, node_counter, new_fixed, forbidden_facilities))
                
                node_counter += 1
                new_forbidden = forbidden_facilities + [branch_var]
                heapq.heappush(queue, (obj_value, node_counter, fixed_facilities, new_forbidden))
        
        return self.best_solution, self.best_value, self.nodes_explored

class CuttingPlaneSolver:
    def __init__(self, problem: FacilityLocationProblem):
        self.problem = problem
        self.max_iterations = 50
        self.cuts_added = 0
        
    def solve_lp_relaxation(self, A_ub_extra=None, b_ub_extra=None) -> Tuple[float, np.ndarray]:
        """Rješava LP relaksaciju s dodatnim cutting plane ograničenjima"""
        n_vars = self.problem.n_facilities + self.problem.n_facilities * self.problem.n_customers
        
        # Ciljna funkcija
        c = np.concatenate([
            self.problem.fixed_costs,
            self.problem.transport_costs.flatten()
        ])
        
        # Osnovna ograničenja
        A_eq = []
        b_eq = []
        
        for j in range(self.problem.n_customers):
            constraint = np.zeros(n_vars)
            for i in range(self.problem.n_facilities):
                xij_index = self.problem.n_facilities + i * self.problem.n_customers + j
                constraint[xij_index] = 1
            A_eq.append(constraint)
            b_eq.append(1)
        
        A_ub = []
        b_ub = []
        
        for i in range(self.problem.n_facilities):
            for j in range(self.problem.n_customers):
                constraint = np.zeros(n_vars)
                constraint[i] = -1
                xij_index = self.problem.n_facilities + i * self.problem.n_customers + j
                constraint[xij_index] = 1
                A_ub.append(constraint)
                b_ub.append(0)
        
        # Dodaj cutting plane ograničenja
        if A_ub_extra is not None:
            A_ub.extend(A_ub_extra)
            b_ub.extend(b_ub_extra)
        
        # Granice
        bounds = [(0, 1)] * n_vars
        
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=bounds, method='highs')
        
        if result.success:
            return result.fun, result.x
        else:
            return float('inf'), None
    
    def generate_gomory_cut(self, solution: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Generira Gomory cut za najfrakcionalniju yi varijablu"""
        y_vars = solution[:self.problem.n_facilities]
        
        # Pronađi najfrakcionalniju varijablu
        max_frac = 0
        frac_var = -1
        
        for i, yi in enumerate(y_vars):
            frac = abs(yi - round(yi))
            if frac > max_frac and frac > 1e-6:
                max_frac = frac
                frac_var = i
        
        if frac_var == -1:
            return None
        
        # Jednostavan Gomory cut: yi <= floor(yi)
        n_vars = len(solution)
        cut_constraint = np.zeros(n_vars)
        cut_constraint[frac_var] = 1
        cut_rhs = math.floor(y_vars[frac_var])
        
        return cut_constraint, cut_rhs
    
    def solve_with_cuts(self) -> Tuple[Optional[np.ndarray], float, int]:
        """Rješava problem koristeći cutting planes"""
        extra_constraints = []
        extra_rhs = []
        
        for iteration in range(self.max_iterations):
            obj_value, solution = self.solve_lp_relaxation(extra_constraints, extra_rhs)
            
            if solution is None:
                break
            
            # Provjeri je li cjelobrojno
            y_vars = solution[:self.problem.n_facilities]
            if all(abs(yi - round(yi)) < 1e-6 for yi in y_vars):
                return solution, obj_value, iteration + 1
            
            # Generiši Gomory cut
            cut = self.generate_gomory_cut(solution)
            if cut is None:
                break
            
            cut_constraint, cut_rhs = cut
            extra_constraints.append(cut_constraint)
            extra_rhs.append(cut_rhs)
            self.cuts_added += 1
        
        # Ako nismo našli cjelobrojno rješenje, pokušaj jednostavno zaokruživanje
        if solution is not None:
            y_vars = solution[:self.problem.n_facilities]
            rounded_y = [round(yi) for yi in y_vars]
            
            # Provjeri je li validno
            total_cost = sum(self.problem.fixed_costs[i] * rounded_y[i] for i in range(self.problem.n_facilities))
            
            # Dodaj transport troškove 
            for j in range(self.problem.n_customers):
                min_cost = float('inf')
                for i in range(self.problem.n_facilities):
                    if rounded_y[i] == 1:
                        min_cost = min(min_cost, self.problem.transport_costs[i, j])
                if min_cost < float('inf'):
                    total_cost += min_cost
            
            return solution, total_cost, self.max_iterations
        
        return None, float('inf'), self.max_iterations

def compare_algorithms():
    """Poredi Branch & Bound i Cutting Plane algoritme"""
    print("\n" + "="*60)
    print("POREĐENJE ALGORITMA")
    print("="*60)
    
    # Test na različitim veličinama problema
    test_cases = [
        (8, 12, "Mali problem"),
        (10, 15, "Srednji problem"),
        (12, 18, "Veliki problem")
    ]
    
    results = []
    
    for n_facilities, n_customers, description in test_cases:
        print(f"\n{description}: {n_facilities} facility, {n_customers} kupaca")
        print("-" * 50)
        
        problem = FacilityLocationProblem(n_facilities, n_customers)
        
        # Branch & Bound
        print("Branch & Bound:")
        bb_solver = BranchAndBoundSolver(problem)
        start_time = time.time()
        bb_solution, bb_value, bb_nodes = bb_solver.branch_and_bound()
        bb_time = time.time() - start_time
        
        print(f"  Vrijeme: {bb_time:.4f}s")
        print(f"  Čvorova istraženo: {bb_nodes}")
        print(f"  Optimalna vrijednost: {bb_value:.2f}")
        
        # Cutting Plane
        print("Cutting Plane:")
        cp_solver = CuttingPlaneSolver(problem)
        start_time = time.time()
        cp_solution, cp_value, cp_iterations = cp_solver.solve_with_cuts()
        cp_time = time.time() - start_time
        
        print(f"  Vrijeme: {cp_time:.4f}s")
        print(f"  Iteracija: {cp_iterations}")
        print(f"  Cutova dodano: {cp_solver.cuts_added}")
        print(f"  Optimalna vrijednost: {cp_value:.2f}")
        
        # Analiza
        speedup = bb_time / cp_time if cp_time > 0 else float('inf')
        
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({
            'description': description,
            'bb_time': bb_time,
            'cp_time': cp_time,
            'bb_nodes': bb_nodes,
            'cp_cuts': cp_solver.cuts_added,
            'speedup': speedup
        })
    
    # Sažetak rezultata
    print(f"\n{'='*60}")
    print("SAŽETAK REZULTATA")
    print(f"{'='*60}")
    print(f"{'Problem':<15} {'B&B Time':<10} {'CP Time':<10} {'Speedup':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['description']:<15} {result['bb_time']:<10.4f} {result['cp_time']:<10.4f} "
              f"{result['speedup']:<10.2f}")
    
    avg_speedup = sum(r['speedup'] for r in results if r['speedup'] != float('inf')) / len(results)
    print(f"\nProječni speedup: {avg_speedup:.2f}x")
    
    if avg_speedup > 1:
        print("🏆 Cutting Plane je brži!")
    else:
        print("🏆 Branch & Bound je brži!")

if __name__ == "__main__":
    compare_algorithms()
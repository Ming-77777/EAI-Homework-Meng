from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

Pos = Tuple[int, int]  # (i,j) with i,j in {1,2,3}

# -------------------------
# Grid helpers
# -------------------------
def all_cells(n: int = 3) -> List[Pos]:
    return [(i, j) for i in range(1, n + 1) for j in range(1, n + 1)]

def adj_4(pos: Pos, n: int = 3) -> List[Pos]:
    i, j = pos
    cand = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
    return [(u, v) for (u, v) in cand if 1 <= u <= n and 1 <= v <= n]

def idx(pos: Pos, n: int = 3) -> int:
    i, j = pos
    return (i - 1) * n + (j - 1)

def bit(mask: int, pos: Pos, n: int = 3) -> bool:
    return ((mask >> idx(pos, n)) & 1) == 1

# -------------------------
# KB + exact inference
# -------------------------
@dataclass(frozen=True)
class Observation:
    pos: Pos
    creaking: bool
    rumbling: bool

@dataclass
class InferenceResult:
    entailed_safe: Dict[Pos, bool]
    entailed_damaged: Dict[Pos, bool]
    entailed_forklift: Dict[Pos, bool]
    classification: Dict[Pos, str]  # SAFE / DANGEROUS / UNKNOWN / INCONSISTENT
    num_models: int

class HazardKB:
    """
    Propositional symbols (conceptually):
      D_ij : damaged floor at (i,j)
      F_ij : forklift at (i,j)
      C_ij : creaking perceived at (i,j)
      R_ij : rumbling perceived at (i,j)
      S_ij : square (i,j) is safe

    Physics:
      C_ij <-> OR_{adj} D
      R_ij <-> OR_{adj} F
      S_ij <-> (~D_ij & ~F_ij)

    Initial knowledge:
      S_11
      at least one D
      at least one F
    """

    def __init__(self, n: int = 3):
        self.n = n
        self.observations: List[Observation] = []
        self.require_at_least_one_damaged = True
        self.require_at_least_one_forklift = True
        self.start_safe: Optional[Pos] = (1, 1)

    # -------- Deliverable A: sentences --------
    def physics_sentences(self) -> List[str]:
        sents: List[str] = []
        for (i, j) in all_cells(self.n):
            adj = adj_4((i, j), self.n)
            disj_d = " ∨ ".join([f"D_{u}{v}" for (u, v) in adj]) if adj else "False"
            disj_f = " ∨ ".join([f"F_{u}{v}" for (u, v) in adj]) if adj else "False"
            sents.append(f"C_{i}{j} ↔ ({disj_d})")
            sents.append(f"R_{i}{j} ↔ ({disj_f})")
            sents.append(f"S_{i}{j} ↔ (¬D_{i}{j} ∧ ¬F_{i}{j})")
        return sents

    def initial_sentences(self) -> List[str]:
        sents: List[str] = []
        if self.start_safe is not None:
            i, j = self.start_safe
            sents.append(f"S_{i}{j}")
        if self.require_at_least_one_damaged:
            sents.append(" ∨ ".join([f"D_{i}{j}" for (i, j) in all_cells(self.n)]))
        if self.require_at_least_one_forklift:
            sents.append(" ∨ ".join([f"F_{i}{j}" for (i, j) in all_cells(self.n)]))
        return sents

    def add_observation(self, pos: Pos, creaking: bool, rumbling: bool) -> None:
        self.observations.append(Observation(pos, creaking, rumbling))

    # -------- model checking --------
    def _satisfies(self, dmask: int, fmask: int) -> bool:
        if self.require_at_least_one_damaged and dmask == 0:
            return False
        if self.require_at_least_one_forklift and fmask == 0:
            return False

        if self.start_safe is not None:
            if bit(dmask, self.start_safe, self.n) or bit(fmask, self.start_safe, self.n):
                return False

        for obs in self.observations:
            adj = adj_4(obs.pos, self.n)
            creak_truth = any(bit(dmask, p, self.n) for p in adj)
            rumble_truth = any(bit(fmask, p, self.n) for p in adj)
            if creak_truth != obs.creaking:
                return False
            if rumble_truth != obs.rumbling:
                return False

        return True

    def models(self) -> Iterable[Tuple[int, int]]:
        nvars = self.n * self.n
        for dmask in range(1 << nvars):
            if self.require_at_least_one_damaged and dmask == 0:
                continue
            for fmask in range(1 << nvars):
                if self.require_at_least_one_forklift and fmask == 0:
                    continue
                if self._satisfies(dmask, fmask):
                    yield (dmask, fmask)

    def entailments(self) -> InferenceResult:
        cells = all_cells(self.n)
        ent_safe = {p: True for p in cells}
        ent_d = {p: True for p in cells}
        ent_f = {p: True for p in cells}
        num = 0

        for (dmask, fmask) in self.models():
            num += 1
            for p in cells:
                d = bit(dmask, p, self.n)
                f = bit(fmask, p, self.n)
                s = (not d) and (not f)
                ent_safe[p] = ent_safe[p] and s
                ent_d[p] = ent_d[p] and d
                ent_f[p] = ent_f[p] and f

        if num == 0:
            return InferenceResult(
                entailed_safe={p: False for p in cells},
                entailed_damaged={p: False for p in cells},
                entailed_forklift={p: False for p in cells},
                classification={p: "INCONSISTENT" for p in cells},
                num_models=0,
            )

        classification: Dict[Pos, str] = {}
        for p in cells:
            if ent_safe[p]:
                classification[p] = "SAFE"
            elif ent_d[p] or ent_f[p]:
                classification[p] = "DANGEROUS"
            else:
                classification[p] = "UNKNOWN"

        return InferenceResult(ent_safe, ent_d, ent_f, classification, num)

# -------------------------
# Plotting: square grid (no files; show figures)
# -------------------------
def show_logic_grid(labels: Dict[Pos, str], title: str, n: int = 3) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    COLOR = {
        "SAFE": (0.72, 0.89, 0.78),        # soft green
        "DANGEROUS": (1.00, 0.70, 0.70),   # soft red
        "UNKNOWN": (0.88, 0.89, 0.91),     # soft gray
        "INCONSISTENT": (1.00, 0.84, 0.65) # soft orange
    }

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.set_aspect("equal")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(range(n + 1))
    ax.set_yticks(range(n + 1))
    ax.grid(True, linewidth=1)
    ax.set_title(title)

    # Keep (1,1) at top-left by mapping y = n - i
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            lab = labels[(i, j)]
            x = j - 1
            y = n - i
            rect = Rectangle(
                (x, y), 1, 1,
                facecolor=COLOR.get(lab, COLOR["UNKNOWN"]),
                edgecolor="black"
            )
            ax.add_patch(rect)
            ax.text(x + 0.5, y + 0.62, lab, ha="center", va="center", fontsize=11, weight="bold")
            ax.text(x + 0.5, y + 0.25, f"({i},{j})", ha="center", va="center", fontsize=10)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.tight_layout()

# -------------------------
# Console helpers
# -------------------------
def fmt_pos_list(ps: List[Pos]) -> str:
    return "[" + ", ".join([f"({i},{j})" for (i, j) in ps]) + "]"

# -------------------------
# Main
# -------------------------
def main() -> None:
    kb = HazardKB(n=3)

    # ==========================
    # [Deliverable A] Parts 1–2
    # ==========================
    print("\n=== [Deliverable A] Part 1: Encode the physics (expanded) ===")
    for s in kb.physics_sentences():
        print(s)

    print("\n=== [Deliverable A] Part 2: Initial knowledge ===")
    for s in kb.initial_sentences():
        print(s)

    # ==========================
    # Part 3: Observation 1
    # ==========================
    kb.add_observation((1, 1), creaking=False, rumbling=False)
    inf1 = kb.entailments()

    print("\n=== [Deliverable B] Part 3: Scenario reasoning ===")
    print("Observation 1: at (1,1), perceive ¬C_11 and ¬R_11")
    print("Adj(1,1) = {(1,2),(2,1)}")
    print("1) C_11 ↔ (D_21 ∨ D_12), ¬C_11 ⟹ ¬D_21 ∧ ¬D_12")
    print("2) R_11 ↔ (F_21 ∨ F_12), ¬R_11 ⟹ ¬F_21 ∧ ¬F_12")
    print("3) S_ij ↔ (¬D_ij ∧ ¬F_ij) ⟹ S_12 and S_21; also given S_11")
    safe1 = sorted([p for p, v in inf1.entailed_safe.items() if v])
    print("Provably safe squares after obs1:", fmt_pos_list(safe1))
    print("Consistency check (#models):", inf1.num_models)

    # [Deliverable C] figure 1
    print("\n=== [Deliverable C] Figure 1: Grid after observation 1 (matplotlib window) ===")
    show_logic_grid(inf1.classification, "After Observation 1 at (1,1): ¬C_11 ∧ ¬R_11", n=3)

    # ==========================
    # Part 4: Observation 2
    # ==========================
    kb.add_observation((2, 1), creaking=True, rumbling=False)
    inf2 = kb.entailments()

    print("\n=== [Deliverable B] Part 4: Exploration ===")
    print("Observation 2: move to (2,1), perceive C_21 and ¬R_21")
    print("Adj(2,1) = {(1,1),(3,1),(2,2)}")
    print("1) R_21 ↔ (F_11 ∨ F_31 ∨ F_22), ¬R_21 ⟹ ¬F_11 ∧ ¬F_31 ∧ ¬F_22")
    print("2) C_21 ↔ (D_11 ∨ D_31 ∨ D_22), C_21 ⟹ (D_11 ∨ D_31 ∨ D_22)")
    print("3) From S_11 ↔ (¬D_11 ∧ ¬F_11) and S_11 ⟹ ¬D_11")
    print("4) Therefore (D_31 ∨ D_22) is entailed")
    safe2 = sorted([p for p, v in inf2.entailed_safe.items() if v])
    print("Provably safe squares after obs2:", fmt_pos_list(safe2))
    print("Consistency check (#models):", inf2.num_models)

    # [Deliverable C] figure 2
    print("\n=== [Deliverable C] Figure 2: Grid after observation 2 (matplotlib window) ===")
    show_logic_grid(inf2.classification, "After Observation 2 at (2,1): C_21 ∧ ¬R_21", n=3)

    # Show both figures
    import matplotlib.pyplot as plt
    plt.show()

if __name__ == "__main__":
    main()

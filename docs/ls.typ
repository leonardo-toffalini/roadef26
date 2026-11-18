#set page(paper: "a4", margin: (x: 2cm, y: 1.8cm), numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.")
#show heading: set block(above: 1em, below: 0.6em)

#align(center)[
  #text(size: 14pt, weight: "bold")[Local search over segment-routing waypoints]
  #v(-0.4em)
  #text(size: 9pt)[`src/ls.py`]
]

= Problem in brief

The network is a directed graph $G = (V, A)$. Every link $a in A$ has a weight $m_a$, which sets its length for shortest paths, and a capacity $c_a$. Time is split into slots $t = 1, dots, T$; in slot $t$ some links are down, which leaves a graph $G_t$. Each demand $d$ goes from $s_d$ to $t_d$ with volume $v_(d,t)$ in slot $t$.

A demand is routed with *segment routing*: we choose a waypoint list $W_(d,t) = (w_1, dots, w_k)$ with $k <= S - 1$. The packet goes $s_d -> w_1 -> dots -> w_k -> t_d$. Each segment follows the shortest paths of $G_t$, with *ECMP*: at every node, the flow splits evenly over all next hops that lie on a shortest path.

The load is measured as *utilization* $u_(a,t) = "traffic"_(a,t) \/ c_a$. Between two consecutive slots the routes can change only within a *budget*: the cost of changing a route is the number of segments that are in one route but not in the other (a symmetric difference), and the total cost at slot $t$ must stay $<= B_t$. The goal is to minimise the utilization vector *lexicographically*: first the largest $u_(a,t)$, then the second largest, and so on.

= Flows are linear in the routes

The key fact: once $G_t$ is fixed, the ECMP flow between two nodes does not depend on any routing decision. So for every ordered pair $(x, y)$ we compute once a *unit-flow vector* $phi^t_(x y) in RR^A$: the fraction of one unit sent from $x$ to $y$ that crosses each arc, already divided by $c_a$. A route is a sum of segments, and utilization is a sum over demands:
$
  u_(dot, t) = sum_d v_(d,t) sum_((x,y) in "seg"(W_(d,t))) phi^t_(x y).
$
Changing the route of one demand only changes its own term $c_(d,t) = v_(d,t) sum phi^t_(x y)$. The rest of the network is untouched.

*Computing $phi$.* Fix a destination $y$. The arcs on some shortest path to $y$ form a DAG. Let $P$ be the ECMP split matrix on it ($P_(x z) = 1 \/ deg_y (x)$ when $(x, z)$ is a next hop). The share of the flow from $s$ that passes node $x$ is
$
  N = I + P + P^2 + dots = (I - P)^(-1).
$
The series is finite because $P$ is nilpotent: a DAG has no cycles. The flow on arc $(x, z)$ is then $N_(s x) \/ deg_y (x)$. One matrix series per destination gives the flows from every source at once. The slots are independent, so this precomputation runs in parallel.

= A smooth surrogate for the lexicographic objective

Minimising the maximum alone gives a flat landscape. Once the hottest arc cannot go lower, almost every move looks neutral. The solver therefore compares states by the pair
$
  (max_(a,t) u_(a,t), quad sum_(a,t) u_(a,t)^16)
$
lexicographically. The second term is the 16th power of an $ell_16$ norm. It is a good proxy for the rest of the lexmin order: an arc at $0.9$ weighs $0.9^16 approx 0.19$ of an arc at $1.0$, and an arc at $0.5$ weighs about $1.5 dot 10^(-5)$. So it still reacts to the hot arcs that are not the maximum and ignores the cold ones. A move that leaves the maximum unchanged but cools the second-, third-, … hottest arcs strictly improves this score. That is what the lexmin objective rewards.

Both parts update *incrementally*. If demand $d$ moves from contribution $c$ to $c'$ in slot $t$, with $b = u_(dot,t) - c$:
- new max $= max(max_a b_a, max_(a in "supp"(c')) (b_a + c'_a))$, because the load only grows on the arcs of the new route;
- new sum $= sum_a b_a^16 + sum_(a in "supp"(c')) ((b_a + c'_a)^16 - b_a^16)$.

So one candidate route is scored in time proportional to its own length, not to $|A|$.

= Neighbourhood <neighbourhood>

A move re-routes *one demand $d$*. The candidate routes are:
+ the direct route (no waypoint);
+ every single waypoint $w in V without {s_d, t_d}$;
+ if no single waypoint improves, two-waypoint routes $(w_1, w_2)$, where $w_1$ is one of the 4 best single waypoints and $w_2$ is any node. This keeps the pair search at $O(|V|)$, not $O(|V|^2)$.

Each candidate is applied in one of two ways: *in one slot $t$*, or *in all slots at once*. The second way matters because of the budget. A route that is the same in every slot costs nothing between slots, so a demand can be moved for the whole horizon at zero budget. A change in one slot costs budget at both of its boundaries, $t$ and $t + 1$. After every move the budget is checked exactly, by recomputing the symmetric-difference cost only for the demand that changed. Among the feasible improving candidates, the move with the best (max, $ell_16$) score is kept (*best improvement inside a demand*).

= Search strategy

The search is driven by *hot spots*. The lexmin objective only cares about the hottest arcs, so it is a waste to try moves elsewhere.
- Take the 30 hottest pairs (arc, slot).
- On each pair, rank the demands by their contribution there and try the 50 largest.
- Apply the *first demand* that has an improving move, then start over from the new hot spots (*first improvement across demands*).
- *Don't-look bits:* a demand with no improving move is skipped from then on. When nothing improves, the bits are cleared once to give every demand another try (other moves may have opened new options). If the next pass also finds nothing, the search has reached a local optimum and stops. It also stops at the time limit.

Every accepted move strictly lowers the (max, $ell_16$) pair, so the search cannot cycle.

= Extension: shortest paths in the closure (`src/ls_closure.py`)

The waypoint scan above tries every node $w$ as a waypoint. The extension asks a single optimization question for each demand instead: which route lowers the score most, to first order?

*Routes are paths in the closure.* Let $H_t$ be the *transitive closure* of $G_t$: a directed graph on the same nodes, with an edge $(u, v)$ whenever $v$ can be reached from $u$ in $G_t$. A closure edge is exactly one segment. A route $s_d -> w_1 -> dots -> w_k -> t_d$ is a path in $H_t$ with $k + 1$ edges, so choosing a waypoint list with at most $S - 1$ waypoints is the same as choosing an $s_d$–$t_d$ path of at most $S$ edges in $H_t$.

*Edge lengths from the gradient.* Let $b = u_(dot,t) - c_(d,t)$ be the load with demand $d$ removed, and $F(u) = sum_a u_a^16$. Sending $d$ along segment $(x, y)$ adds $v_(d,t) phi^t_(x y)$, so to first order
$
  F(b + v_(d,t) phi^t_(x y)) approx F(b) + v_(d,t) sum_a 16 b_a^15 phi^t_(x y, a).
$
This gives every closure edge a length (metricizes the closure):
$
  lambda_(x y) = sum_a 16 b_a^15 phi^t_(x y, a) >= 0.
$
The approximation is linear, so the cost of a route is the sum of its edge lengths, and the best route under it is a *shortest path* in $(H_t, lambda)$. The volume $v_(d,t) > 0$ multiplies every length by the same constant, so it does not change the path. The weight $b_a^15$ makes an arc at load $0.9$ cost about $0.2$ times an arc at $1.0$, and empty arcs cost nothing. In effect the metric says "stay away from the hottest arcs". This is the linear step of the Frank–Wolfe method, restricted to the routes segment routing allows.

*Hop-bounded shortest path.* The number of edges is bounded, so the solver runs $k$ rounds of Bellman–Ford: $D_k (v) = min_u (D_(k-1)(u) + lambda_(u v))$, where $s_d$ may only start the path and $t_d$ may only end it. On a tie, fewer edges wins, so a free detour is never added. By default at most 2 edges are allowed (one waypoint). Then the path is just $min(lambda_(s t), min_w lambda_(s w) + lambda_(w t))$, and only $2|V|$ edge lengths are needed instead of all $|V|^2$.

*Exact check.* The linearization ignores the demand's own load and the max term, so the path is only a proposal. It is accepted only if the exact (max, $ell_16$) score improves and the budget holds, applied either in one slot or in all slots, as in @neighbourhood. The search loop is the same, but each (demand, slot) pair now tests one route instead of $|V|$, and the don't-look bits are kept per (demand, slot).

= Limits and possible extensions

- *Local optimum.* Only single-demand moves exist. A hot arc that can only be relieved by moving two demands at once stays stuck. Moves that swap or shift two demands together, or a perturbation step (iterated local search), would help here.
- *Greedy budget use.* The budget goes to the first moves that improve. Early, small gains can use up the budget that a later, larger gain would need. The "all slots" move avoids part of this problem, but not all of it.
- *At most two waypoints.* Routes can use up to $S - 1$ waypoints, but the scan stops at two. The closure search can go further by allowing more edges, but by default it also stops at one waypoint.
- *The surrogate is not lexmin.* The $ell_16$ term only approximates the lexicographic order. Two states with the same (max, $ell_16$) score can still have different lexmin rank. The exponent sets how far below the maximum the search still "sees" an arc.

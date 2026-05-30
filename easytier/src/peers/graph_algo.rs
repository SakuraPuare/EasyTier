use core::cmp::Ordering;
use petgraph::{
    algo::Measure,
    visit::{EdgeRef, IntoEdges, VisitMap, Visitable},
};
use std::collections::HashMap;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::{collections::BinaryHeap, hash::Hash};

/// `MinScored<K, T>` holds a score `K` and a scored object `T` in
/// a pair for use with a `BinaryHeap`.
///
/// `MinScored` compares in reverse order by the score, so that we can
/// use `BinaryHeap` as a min-heap to extract the score-value pair with the
/// least score.
///
/// **Note:** `MinScored` implements a total order (`Ord`), so that it is
/// possible to use float types as scores.
#[derive(Copy, Clone, Debug)]
pub struct MinScored<K, T>(pub K, pub T);

impl<K: PartialOrd, T> PartialEq for MinScored<K, T> {
    #[inline]
    fn eq(&self, other: &MinScored<K, T>) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<K: PartialOrd, T> Eq for MinScored<K, T> {}

impl<K: PartialOrd, T> PartialOrd for MinScored<K, T> {
    #[inline]
    fn partial_cmp(&self, other: &MinScored<K, T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: PartialOrd, T> Ord for MinScored<K, T> {
    #[inline]
    fn cmp(&self, other: &MinScored<K, T>) -> Ordering {
        let a = &self.0;
        let b = &other.0;
        if a == b {
            Ordering::Equal
        } else if a < b {
            Ordering::Greater
        } else if a > b {
            Ordering::Less
        } else if a.ne(a) && b.ne(b) {
            // these are the NaN cases
            Ordering::Equal
        } else if a.ne(a) {
            // Order NaN less, so that it is last in the MinScore order
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}

pub type DijkstraResult<K, NodeId> = (HashMap<NodeId, K>, HashMap<NodeId, (NodeId, usize)>);

pub fn dijkstra_with_first_hop<G, F, K>(
    graph: G,
    start: G::NodeId,
    mut edge_cost: F,
) -> DijkstraResult<K, G::NodeId>
where
    G: IntoEdges + Visitable,
    G::NodeId: Eq + Hash + Clone,
    F: FnMut(G::EdgeRef) -> K,
    K: Measure + Copy,
{
    let mut visited = graph.visit_map();
    let mut scores = HashMap::new();
    let mut first_hop = HashMap::new();
    let mut visit_next = BinaryHeap::new();
    let zero_score = K::default();
    scores.insert(start, zero_score);
    visit_next.push(MinScored(zero_score, start));
    first_hop.insert(start, (start, 0));

    while let Some(MinScored(node_score, node)) = visit_next.pop() {
        if visited.is_visited(&node) {
            continue;
        }
        for edge in graph.edges(node) {
            let next = edge.target();
            if visited.is_visited(&next) {
                continue;
            }
            let next_score = node_score + edge_cost(edge);
            match scores.entry(next) {
                Occupied(mut ent) => {
                    if next_score < *ent.get() {
                        *ent.get_mut() = next_score;
                        visit_next.push(MinScored(next_score, next));
                        // 继承前驱的 first_hop，或自己就是第一跳
                        let hop = if node == start {
                            (next, 0)
                        } else {
                            first_hop[&node]
                        };
                        first_hop.insert(next, (hop.0, hop.1 + 1));
                    }
                }
                Vacant(ent) => {
                    ent.insert(next_score);
                    visit_next.push(MinScored(next_score, next));
                    let hop = if node == start {
                        (next, 0)
                    } else {
                        first_hop[&node]
                    };
                    first_hop.insert(next, (hop.0, hop.1 + 1));
                }
            }
        }
        visited.visit(node);
    }

    (scores, first_hop)
}

/// Like `dijkstra_with_first_hop`, but collects ALL first-hops with equal minimal cost
/// for each destination node. This enables ECMP-style multi-path routing.
pub type DijkstraMultiHopResult<K, NodeId> = (HashMap<NodeId, K>, HashMap<NodeId, Vec<(NodeId, usize)>>);

pub fn dijkstra_with_all_first_hops<G, F, K>(
    graph: G,
    start: G::NodeId,
    mut edge_cost: F,
) -> DijkstraMultiHopResult<K, G::NodeId>
where
    G: IntoEdges + Visitable,
    G::NodeId: Eq + Hash + Clone,
    F: FnMut(G::EdgeRef) -> K,
    K: Measure + Copy,
{
    let mut visited = graph.visit_map();
    let mut scores = HashMap::new();
    let mut first_hops: HashMap<G::NodeId, Vec<(G::NodeId, usize)>> = HashMap::new();
    let mut visit_next = BinaryHeap::new();
    let zero_score = K::default();
    scores.insert(start, zero_score);
    visit_next.push(MinScored(zero_score, start));
    first_hops.insert(start, vec![(start, 0)]);

    while let Some(MinScored(node_score, node)) = visit_next.pop() {
        if visited.is_visited(&node) {
            continue;
        }
        for edge in graph.edges(node) {
            let next = edge.target();
            if visited.is_visited(&next) {
                continue;
            }
            let next_score = node_score + edge_cost(edge);
            match scores.entry(next) {
                Occupied(mut ent) => {
                    if next_score < *ent.get() {
                        // Found a better path, replace
                        *ent.get_mut() = next_score;
                        visit_next.push(MinScored(next_score, next));
                        let hop = if node == start {
                            (next, 0)
                        } else {
                            // Inherit first_hops from predecessor
                            first_hops.get(&node).and_then(|h| h.first().copied()).unwrap_or((next, 0))
                        };
                        first_hops.insert(next, vec![(hop.0, hop.1 + 1)]);
                    } else if next_score == *ent.get() {
                        // Found an equal-cost path, add as alternative
                        let hop = if node == start {
                            (next, 0)
                        } else {
                            first_hops.get(&node).and_then(|h| h.first().copied()).unwrap_or((next, 0))
                        };
                        let new_first_hop = (hop.0, hop.1 + 1);
                        first_hops.entry(next).or_default().push(new_first_hop);
                    }
                }
                Vacant(ent) => {
                    ent.insert(next_score);
                    visit_next.push(MinScored(next_score, next));
                    let hop = if node == start {
                        (next, 0)
                    } else {
                        first_hops.get(&node).and_then(|h| h.first().copied()).unwrap_or((next, 0))
                    };
                    first_hops.insert(next, vec![(hop.0, hop.1 + 1)]);
                }
            }
        }
        visited.visit(node);
    }

    (scores, first_hops)
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::DiGraph;

    #[test]
    fn test_dijkstra_with_first_hop_4node() {
        let mut graph = DiGraph::<&str, u32>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let d = graph.add_node("d");

        graph.extend_with_edges([(a, b, 1)]);
        graph.extend_with_edges([(b, c, 1)]);
        graph.extend_with_edges([(c, d, 2)]);

        let (scores, first_hop) = dijkstra_with_first_hop(&graph, a, |edge| *edge.weight());

        assert_eq!(scores[&b], 1);
        assert_eq!(scores[&c], 2);
        assert_eq!(scores[&d], 4);

        assert_eq!(first_hop[&b], (b, 1));
        assert_eq!(first_hop[&c], (b, 2));
        assert_eq!(first_hop[&d], (b, 3));
    }

    #[test]
    fn test_dijkstra_with_first_hop() {
        let mut graph = DiGraph::<&str, u32>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let d = graph.add_node("d");
        let e = graph.add_node("e");

        graph.extend_with_edges([(a, b, 1), (a, c, 2), (b, d, 1), (c, d, 3), (d, e, 1)]);

        let (scores, first_hop) = dijkstra_with_first_hop(&graph, a, |edge| *edge.weight());

        assert_eq!(scores[&b], 1);
        assert_eq!(scores[&c], 2);
        assert_eq!(scores[&d], 2);
        assert_eq!(scores[&e], 3);

        assert_eq!(first_hop[&b], (b, 1));
        assert_eq!(first_hop[&c], (c, 1));
        assert_eq!(first_hop[&d], (b, 2)); // d is reached via b
        assert_eq!(first_hop[&e], (b, 3)); // e is reached via d
    }

    #[test]
    fn test_dijkstra_with_all_first_hops_diamond() {
        // Diamond topology: a -> b -> d, a -> c -> d (both paths cost 2)
        let mut graph = DiGraph::<&str, u32>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let d = graph.add_node("d");

        graph.extend_with_edges([(a, b, 1), (a, c, 1), (b, d, 1), (c, d, 1)]);

        let (scores, first_hops) = dijkstra_with_all_first_hops(&graph, a, |edge| *edge.weight());

        assert_eq!(scores[&a], 0);
        assert_eq!(scores[&b], 1);
        assert_eq!(scores[&c], 1);
        assert_eq!(scores[&d], 2);

        // b and c are direct neighbors, each has one first hop
        assert_eq!(first_hops[&b].len(), 1);
        assert_eq!(first_hops[&b][0].0, b);
        assert_eq!(first_hops[&c].len(), 1);
        assert_eq!(first_hops[&c][0].0, c);

        // d has two equal-cost first hops: b and c
        assert_eq!(first_hops[&d].len(), 2);
        let hop_ids: Vec<_> = first_hops[&d].iter().map(|(id, _)| *id).collect();
        assert!(hop_ids.contains(&b));
        assert!(hop_ids.contains(&c));
    }

    #[test]
    fn test_dijkstra_with_all_first_hops_asymmetric() {
        // Asymmetric costs: only one path is optimal
        let mut graph = DiGraph::<&str, u32>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let d = graph.add_node("d");

        // a->b costs 1, a->c costs 10, b->d costs 1, c->d costs 1
        graph.extend_with_edges([(a, b, 1), (a, c, 10), (b, d, 1), (c, d, 1)]);

        let (scores, first_hops) = dijkstra_with_all_first_hops(&graph, a, |edge| *edge.weight());

        assert_eq!(scores[&d], 2); // a -> b -> d = 2
        // Only b is the optimal first hop (a->c->d would cost 11)
        assert_eq!(first_hops[&d].len(), 1);
        assert_eq!(first_hops[&d][0].0, b);
    }

    #[test]
    fn test_dijkstra_with_all_first_hops_single_path() {
        // Linear topology: a -> b -> c -> d (only one path)
        let mut graph = DiGraph::<&str, u32>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let d = graph.add_node("d");

        graph.extend_with_edges([(a, b, 1), (b, c, 1), (c, d, 1)]);

        let (scores, first_hops) = dijkstra_with_all_first_hops(&graph, a, |edge| *edge.weight());

        assert_eq!(scores[&d], 3);
        assert_eq!(first_hops[&d].len(), 1);
        assert_eq!(first_hops[&d][0].0, b);
        assert_eq!(first_hops[&d][0].1, 3); // path length = 3
    }

    #[test]
    fn test_dijkstra_with_all_first_hops_multiple_equal_cost_intermediate() {
        // a -> b -> d (cost 3), a -> c -> d (cost 3), but b and c have different intermediate costs
        // This tests that equal total cost paths are both collected
        let mut graph = DiGraph::<&str, u32>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let d = graph.add_node("d");

        // a->b costs 2, a->c costs 1, b->d costs 1, c->d costs 2
        // Both paths a->b->d and a->c->d cost 3
        graph.extend_with_edges([(a, b, 2), (a, c, 1), (b, d, 1), (c, d, 2)]);

        let (scores, first_hops) = dijkstra_with_all_first_hops(&graph, a, |edge| *edge.weight());

        assert_eq!(scores[&d], 3);
        // Both b and c should be first hops for d (equal total cost)
        assert_eq!(first_hops[&d].len(), 2);
        let hop_ids: Vec<_> = first_hops[&d].iter().map(|(id, _)| *id).collect();
        assert!(hop_ids.contains(&b));
        assert!(hop_ids.contains(&c));
    }
}

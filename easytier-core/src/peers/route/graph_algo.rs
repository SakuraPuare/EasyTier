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

/// Result of the multi-first-hop dijkstra: for each destination, all equal-cost first hops.
/// Each entry is (first_hop_node, path_length).
pub type DijkstraMultiHopResult<K, NodeId> =
    (HashMap<NodeId, K>, HashMap<NodeId, Vec<(NodeId, usize)>>);

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

/// Like `dijkstra_with_first_hop`, but collects ALL equal-cost first hops per destination (ECMP).
/// When multiple paths to a destination have the same total cost, all distinct first-hop nodes
/// are preserved. This enables multi-relay load balancing.
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
    let mut scores: HashMap<G::NodeId, K> = HashMap::new();
    let mut first_hops: HashMap<G::NodeId, Vec<(G::NodeId, usize)>> = HashMap::new();
    let mut visit_next = BinaryHeap::new();
    let zero_score = K::default();
    scores.insert(start.clone(), zero_score);
    visit_next.push(MinScored(zero_score, start.clone()));

    while let Some(MinScored(node_score, node)) = visit_next.pop() {
        if visited.is_visited(&node) {
            continue;
        }
        for edge in graph.edges(node.clone()) {
            let next = edge.target();
            if visited.is_visited(&next) {
                continue;
            }
            let next_score = node_score + edge_cost(edge);
            // Compute the first-hops that `next` would inherit from `node`
            let inherited: Vec<(G::NodeId, usize)> = if node == start {
                vec![(next.clone(), 1)]
            } else {
                first_hops
                    .get(&node)
                    .map(|hops| {
                        hops.iter()
                            .map(|(h, len)| (h.clone(), len + 1))
                            .collect()
                    })
                    .unwrap_or_default()
            };

            match scores.entry(next.clone()) {
                Occupied(ent) => {
                    if next_score < *ent.get() {
                        // Strictly better path: replace
                        *scores.get_mut(&next).unwrap() = next_score;
                        first_hops.insert(next.clone(), inherited);
                        visit_next.push(MinScored(next_score, next));
                    } else if next_score == *ent.get() {
                        // Equal cost: merge first-hops (deduplicate by hop node id)
                        let existing = first_hops.entry(next).or_default();
                        for hop in inherited {
                            if !existing.iter().any(|(h, _)| *h == hop.0) {
                                existing.push(hop);
                            }
                        }
                    }
                }
                Vacant(ent) => {
                    ent.insert(next_score);
                    first_hops.insert(next.clone(), inherited);
                    visit_next.push(MinScored(next_score, next));
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
    fn test_multi_hop_diamond() {
        // A --1--> B --1--> D
        // A --1--> C --1--> D
        // Both paths to D cost 2; both B and C should be first-hops for D
        let mut graph = DiGraph::<&str, u32>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let d = graph.add_node("d");
        graph.extend_with_edges([(a, b, 1), (a, c, 1), (b, d, 1), (c, d, 1)]);

        let (scores, first_hops) =
            dijkstra_with_all_first_hops(&graph, a, |edge| *edge.weight());

        assert_eq!(scores[&d], 2);
        let mut hops: Vec<_> = first_hops[&d].iter().map(|(n, _)| *n).collect();
        hops.sort();
        let mut expected = vec![b, c];
        expected.sort();
        assert_eq!(hops, expected);
    }

    #[test]
    fn test_multi_hop_asymmetric() {
        // A --1--> B --1--> D
        // A --1--> C --5--> D
        // Only B should be first-hop for D (cost 2 < 6)
        let mut graph = DiGraph::<&str, u32>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let d = graph.add_node("d");
        graph.extend_with_edges([(a, b, 1), (a, c, 1), (b, d, 1), (c, d, 5)]);

        let (_scores, first_hops) =
            dijkstra_with_all_first_hops(&graph, a, |edge| *edge.weight());

        assert_eq!(first_hops[&d].len(), 1);
        assert_eq!(first_hops[&d][0].0, b);
    }

    #[test]
    fn test_multi_hop_propagates_past_branch() {
        // A --1--> B --1--> M --1--> D
        // A --1--> C --1--> M --1--> D
        // D should have both B and C as first-hops (via M)
        let mut graph = DiGraph::<&str, u32>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let m = graph.add_node("m");
        let d = graph.add_node("d");
        graph.extend_with_edges([(a, b, 1), (a, c, 1), (b, m, 1), (c, m, 1), (m, d, 1)]);

        let (scores, first_hops) =
            dijkstra_with_all_first_hops(&graph, a, |edge| *edge.weight());

        assert_eq!(scores[&d], 3);
        let mut hops: Vec<_> = first_hops[&d].iter().map(|(n, _)| *n).collect();
        hops.sort();
        let mut expected = vec![b, c];
        expected.sort();
        assert_eq!(hops, expected);
    }

    #[test]
    fn test_multi_hop_single_path() {
        // A --1--> B --1--> C (no alternatives)
        let mut graph = DiGraph::<&str, u32>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        graph.extend_with_edges([(a, b, 1), (b, c, 1)]);

        let (_scores, first_hops) =
            dijkstra_with_all_first_hops(&graph, a, |edge| *edge.weight());

        assert_eq!(first_hops[&b].len(), 1);
        assert_eq!(first_hops[&b][0].0, b);
        assert_eq!(first_hops[&c].len(), 1);
        assert_eq!(first_hops[&c][0].0, b);
    }
}

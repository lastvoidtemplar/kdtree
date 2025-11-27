use std::{
    collections::BinaryHeap,
    fmt::{Debug, Display},
};

#[derive(Debug)]
struct Node<T> {
    value: T,
    left: Option<Box<Node<T>>>,
    right: Option<Box<Node<T>>>,
}

struct PairDistanceValue<T> {
    value: T,
    dist: f64,
}

impl<T> PartialEq for PairDistanceValue<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl<T> PartialOrd for PairDistanceValue<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl<T> Eq for PairDistanceValue<T> {}

impl<T> Ord for PairDistanceValue<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

pub struct KDTree<T, D, FMap, FDist, FRadius>
where
    T: Clone,
    D: PartialOrd,
    FMap: Fn(&T) -> D,
    FDist: Fn(&T, &T) -> f64,
    FRadius: Fn(&D, &D) -> f64,
{
    root: Option<Box<Node<T>>>,
    dimensions: Vec<FMap>,
    dist_func: FDist,
    radius_func: FRadius,
}

impl<T, D, FMap, FDist, FRadius> KDTree<T, D, FMap, FDist, FRadius>
where
    T: Clone,
    D: PartialOrd,
    FMap: Fn(&T) -> D,
    FDist: Fn(&T, &T) -> f64,
    FRadius: Fn(&D, &D) -> f64,
{
    pub fn new(
        data: &mut [T],
        dimensions: Vec<FMap>,
        dist_func: FDist,
        radius_func: FRadius,
    ) -> Self {
        Self {
            root: Self::build_node(data, &dimensions, 0),
            dimensions,
            dist_func,
            radius_func,
        }
    }

    fn sort_data_by_dimension<'a>(data: &'a mut [T], dimension: &FMap) {
        data.sort_by(|a, b| dimension(a).partial_cmp(&dimension(b)).unwrap_or(std::cmp::Ordering::Equal));
    }

    fn build_node(
        data: &mut [T],
        dimensions: &Vec<FMap>,
        dimension_ind: usize,
    ) -> Option<Box<Node<T>>> {
        if data.is_empty() {
            return None;
        }

        Self::sort_data_by_dimension(data, &dimensions[dimension_ind]);
        let median_ind = data.len() / 2;
        let median = data[median_ind].clone();

        let next_dimension_ind = (dimension_ind + 1) % dimensions.len();

        Some(Box::new(Node {
            value: median,
            left: Self::build_node(&mut data[..median_ind], dimensions, next_dimension_ind),
            right: Self::build_node(&mut data[median_ind + 1..], dimensions, next_dimension_ind),
        }))
    }

    pub fn find_k_nearest_neighbors(&self, target: T, k: usize) -> Vec<T> {
        let mut heap = BinaryHeap::new();
        self.knn(&target, k, &self.root, 0, &mut heap);
        let mut pairs = heap.into_vec();
        pairs.sort_by(|a,b|a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
        pairs.iter().map(|p| p.value.clone()).collect()
    }

    fn knn(
        &self,
        target: &T,
        k: usize,
        node: &Option<Box<Node<T>>>,
        dimension_ind: usize,
        heap: &mut BinaryHeap<PairDistanceValue<T>>,
    ) {
        match node {
            None => return,
            Some(node) => {
                let value = &node.value;
                let dist = (self.dist_func)(target, value);

                if heap.len() < k {
                    heap.push(PairDistanceValue {
                        value: value.clone(),
                        dist,
                    });
                } else if heap.peek().is_some_and(|max| dist < max.dist) {
                    heap.pop();
                    heap.push(PairDistanceValue {
                        value: value.clone(),
                        dist,
                    });
                }

                let dimension = &self.dimensions[dimension_ind];
                let (near, far) = if dimension(target) < dimension(value) {
                    (&node.left, &node.right)
                } else {
                    (&node.right, &node.left)
                };

                let new_dimension_ind = (dimension_ind + 1) % self.dimensions.len();
                self.knn(target, k, near, new_dimension_ind, heap);

                if heap.len() < k
                    || heap.peek().is_some_and(|max| {
                        (self.radius_func)(&dimension(target), &dimension(value)) < max.dist
                    })
                {
                    self.knn(target, k, far, new_dimension_ind, heap);
                }
            }
        }
    }
}

impl<T, D, FMap, FDist, FRadius> Display for KDTree<T, D, FMap, FDist, FRadius>
where
    T: Clone + Debug,
    D: PartialOrd + Debug,
    FMap: Fn(&T) -> D,
    FDist: Fn(&T, &T) -> f64,
    FRadius: Fn(&D, &D) -> f64,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.root)
    }
}

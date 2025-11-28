use std::{collections::BinaryHeap, fmt::{Debug, Display}};

#[derive(Debug)]
struct Node<T> {
    value: T,
    left: Option<Box<Node<T>>>,
    right: Option<Box<Node<T>>>,
}

struct PairDistanceValue<T, D>
where
    D: PartialOrd + PartialEq,
{
    value: T,
    dist: D,
}

impl<T, D> PartialEq for PairDistanceValue<T, D>
where
    D: PartialOrd + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl<T, D> PartialOrd for PairDistanceValue<T, D>
where
    D: PartialOrd + PartialEq,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl<T, D> Eq for PairDistanceValue<T, D> where D: PartialOrd + PartialEq {}

impl<T, D> Ord for PairDistanceValue<T, D>
where
    D: PartialOrd + PartialEq,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

pub struct KDTree<T, D, FMap, FDist, FRadius>
where
    T: Clone,
    D: PartialOrd,
    FMap: Fn(&T) -> D,
    FDist: Fn(&T, &T) -> D,
    FRadius: Fn(&D, &D) -> D,
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
    FDist: Fn(&T, &T) -> D,
    FRadius: Fn(&D, &D) -> D,
{
    pub fn new(
        mut data: Vec<T>,
        dimensions: Vec<FMap>,
        dist_func: FDist,
        radius_func: FRadius,
    ) -> Self {
        Self {
            root: Self::build_node(&mut data, &dimensions, 0),
            dimensions,
            dist_func,
            radius_func,
        }
    }

    fn partition(data: &mut [T], dimension: &FMap) -> usize {
        let len = data.len();
        let pivot = dimension(&data[len - 1]);
        let mut i = 0;
        for j in 0..(len - 1) {
            if dimension(&data[j]) <= pivot {
                data.swap(i, j);
                i += 1;
            }
        }
        data.swap(i, len - 1);
        i
    }

    fn quick_selection<'a>(data: &'a mut [T], dimension: &FMap, ind: usize) -> &'a T {
        if data.len() == 1 {
            return &data[0];
        }

        let pivot_ind = Self::partition(data, dimension);
        if ind == pivot_ind {
            &data[ind]
        } else if ind < pivot_ind {
            Self::quick_selection(&mut data[..pivot_ind], dimension, ind)
        } else {
            Self::quick_selection(&mut data[pivot_ind + 1..], dimension, ind - pivot_ind - 1)
        }
    }

    // fn sort_data_by_dimension(data: &mut [T], dimension: &FMap) {
    //     data.sort_by(|a, b| {
    //         dimension(a)
    //             .partial_cmp(&dimension(b))
    //             .unwrap_or(std::cmp::Ordering::Equal)
    //     });
    // }

    fn build_node(
        data: &mut [T],
        dimensions: &Vec<FMap>,
        dimension_ind: usize,
    ) -> Option<Box<Node<T>>> {
        if data.is_empty() {
            return None;
        }

        let median_ind = data.len() / 2;
        let median = Self::quick_selection(data, &dimensions[dimension_ind], median_ind).clone();

        let next_dimension_ind = (dimension_ind + 1) % dimensions.len();

        Some(Box::new(Node {
            value: median,
            left: Self::build_node(&mut data[..median_ind], dimensions, next_dimension_ind),
            right: Self::build_node(&mut data[median_ind + 1..], dimensions, next_dimension_ind),
        }))
    }

    pub fn find_k_nearest_neighbors(&self, target: &T, k: usize) -> Vec<T> {
        let mut heap = BinaryHeap::new();
        self.knn(target, k, &self.root, 0, &mut heap);
        let mut pairs = heap.into_vec();
        pairs.sort_by(|a, b| {
            a.dist
                .partial_cmp(&b.dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        pairs.iter().map(|p| p.value.clone()).collect()
    }

    fn knn(
        &self,
        target: &T,
        k: usize,
        node: &Option<Box<Node<T>>>,
        dimension_ind: usize,
        heap: &mut BinaryHeap<PairDistanceValue<T, D>>,
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
    FDist: Fn(&T, &T) -> D,
    FRadius: Fn(&D, &D) -> D,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.root)
    }
}
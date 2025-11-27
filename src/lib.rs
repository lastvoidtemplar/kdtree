mod kdtree;

#[derive(Clone, Debug)]
struct Point {
    x: f64,
    y: f64,
}

#[cfg(test)]
mod tests {
    use crate::{Point, kdtree};

    #[test]
    fn test_kdtree_new() {
        let mut data = [
            Point { x: 615.0, y: 40.0 },
            Point { x: 207.0, y: 313.0 },
            Point { x: 751.0, y: 177.0 },
            Point { x: 479.0, y: 449.0 },
            Point { x: 70.0, y: 721.0 },
            Point { x: 343.0, y: 858.0 },
            Point { x: 888.0, y: 585.0 },
        ];

        let dimensions = vec![|p: &Point| p.x, |p: &Point| p.y];

        let dist_func = |p1: &Point, p2: &Point| {
            ((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)).sqrt()
        };

        let radius_func = |d1: &f64, d2: &f64| (d1 - d2).abs();

        let new_kdtree = kdtree::KDTree::new(&mut data, dimensions, dist_func, radius_func);

        println!("{}", new_kdtree)
    }

    #[test]
    fn test_kdtree_find_k_nearest_neighbors_1() {
        let mut data = [
            Point { x: 272.0, y: 59.0 },
            Point { x: 481.0, y: 144.0 },
            Point { x: 915.0, y: 157.0 },
            Point { x: 259.0, y: 189.0 },
            Point { x: 913.0, y: 276.0 },
            Point { x: 139.0, y: 310.0 },
            Point { x: 821.0, y: 386.0 },
            Point { x: 622.0, y: 410.0 },
            Point { x: 281.0, y: 467.0 },
            Point { x: 43.0, y: 480.0 },
            Point { x: 445.0, y: 585.0 },
            Point { x: 136.0, y: 615.0 },
            Point { x: 749.0, y: 683.0 },
            Point { x: 260.0, y: 685.0 },
            Point { x: 592.0, y: 715.0 },
            Point { x: 662.0, y: 798.0 },
            Point { x: 879.0, y: 810.0 },
            Point { x: 163.0, y: 826.0 },
            Point { x: 438.0, y: 828.0 },
            Point { x: 571.0, y: 839.0 },
        ];

        let dimensions = vec![|p: &Point| p.x, |p: &Point| p.y];

        let dist_func = |p1: &Point, p2: &Point| {
            ((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)).sqrt()
        };

        let radius_func = |d1: &f64, d2: &f64| (d1 - d2).abs();

        let tree = kdtree::KDTree::new(&mut data, dimensions, dist_func, radius_func);

        let neighbors = tree.find_k_nearest_neighbors(Point { x: 782.0, y: 780.0 }, 1);

        println!("{:?}", neighbors)
    }

    #[test]
    fn test_kdtree_find_k_nearest_neighbors_2() {
        let mut data = [
            Point { x: 272.0, y: 59.0 },
            Point { x: 481.0, y: 144.0 },
            Point { x: 915.0, y: 157.0 },
            Point { x: 259.0, y: 189.0 },
            Point { x: 913.0, y: 276.0 },
            Point { x: 139.0, y: 310.0 },
            Point { x: 821.0, y: 386.0 },
            Point { x: 622.0, y: 410.0 },
            Point { x: 281.0, y: 467.0 },
            Point { x: 43.0, y: 480.0 },
            Point { x: 445.0, y: 585.0 },
            Point { x: 136.0, y: 615.0 },
            Point { x: 749.0, y: 683.0 },
            Point { x: 260.0, y: 685.0 },
            Point { x: 592.0, y: 715.0 },
            Point { x: 662.0, y: 798.0 },
            Point { x: 879.0, y: 810.0 },
            Point { x: 163.0, y: 826.0 },
            Point { x: 438.0, y: 828.0 },
            Point { x: 571.0, y: 839.0 },
        ];

        let dimensions = vec![|p: &Point| p.x, |p: &Point| p.y];

        let dist_func = |p1: &Point, p2: &Point| {
            ((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)).sqrt()
        };

        let radius_func = |d1: &f64, d2: &f64| (d1 - d2).abs();

        let tree = kdtree::KDTree::new(&mut data, dimensions, dist_func, radius_func);

        let neighbors = tree.find_k_nearest_neighbors(Point { x: 260.0, y: 585.0 }, 5);

        println!("{:?}", neighbors)
    }
}

#[cfg(test)]
mod tests {
    use crate::math::matrix::Matrix;

    #[test]
    fn test_zeros_flat() {
        let m = Matrix::zeros(2, 3);
        assert_eq!(m.data.len(), 6);
        assert!(m.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_from_data_flatten() {
        let m = Matrix::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 2);
        assert_eq!(m.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_transpose() {
        let m = Matrix::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        // Expected: [[1,4],[2,5],[3,6]]
        assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_matmul() {
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Matrix::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::from_data(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let c = Matrix::matmul(&a, &b);
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_add_assign() {
        let mut a = Matrix::from_data(vec![vec![1.0, 2.0]]);
        let b = Matrix::from_data(vec![vec![3.0, 4.0]]);
        a += b;
        assert_eq!(a.data, vec![4.0, 6.0]);
    }

    #[test]
    fn test_sub_assign() {
        let mut a = Matrix::from_data(vec![vec![5.0, 6.0]]);
        let b = Matrix::from_data(vec![vec![1.0, 2.0]]);
        a -= b;
        assert_eq!(a.data, vec![4.0, 4.0]);
    }

    #[test]
    fn test_serde_roundtrip() {
        let original = Matrix::from_data(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let json = serde_json::to_string(&original).unwrap();
        let loaded: Matrix = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.rows, original.rows);
        assert_eq!(loaded.cols, original.cols);
        assert_eq!(loaded.data, original.data);
    }

    #[test]
    fn test_serde_legacy_format() {
        // Simulate a JSON file saved by the old Vec<Vec<f64>> format.
        let legacy_json = r#"{"rows":2,"cols":3,"data":[[1.0,2.0,3.0],[4.0,5.0,6.0]]}"#;
        let m: Matrix = serde_json::from_str(legacy_json).unwrap();
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_map() {
        let m = Matrix::from_data(vec![vec![1.0, 2.0, 3.0]]);
        let doubled = m.map(|x| x * 2.0);
        assert_eq!(doubled.data, vec![2.0, 4.0, 6.0]);
    }
}

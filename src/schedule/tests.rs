use super::*;

#[test]
fn schedule() {
    use Schedule::*;
    assert_eq!(
        (0..8)
            .map(|i| Schedule::real_idx(&Cyclic, i, 8, 3))
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 0, 1, 2, 0, 1]
    );
    assert_eq!(
        (0..8)
            .map(|i| Schedule::real_idx(&Stretched, i, 8, 3))
            .collect::<Vec<_>>(),
        vec![0, 0, 0, 1, 1, 1, 2, 2]
    );
    let custom = vec![0, 1, 2, 2, 1, 0, 0, 0];
    assert_eq!(
        (0..8)
            .map(|i| Schedule::real_idx(&Custom(custom.clone()), i, 8, 3))
            .collect::<Vec<_>>(),
        custom
    );
}

#[test]
fn bidi_schedule() {
    use BidiSchedule::*;
    assert_eq!(
        (0..10)
            .map(|i| BidiSchedule::real_idx(&StridedCyclic, i, 10, 4))
            .collect::<Vec<_>>(),
        vec![
            0, 1, /**/ 2, 3, /**/ 0, 1, /**/ 2, 3, /**/ 0, 1
        ]
    );
    assert_eq!(
        (0..10)
            .map(|i| BidiSchedule::real_idx(&StridedStretched, i, 10, 4))
            .collect::<Vec<_>>(),
        vec![
            0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 2, 3, /**/ 2, 3
        ]
    );
    assert_eq!(
        (0..10)
            .map(|i| BidiSchedule::real_idx(&SymmetricCyclic, i, 10, 4))
            .collect::<Vec<_>>(),
        vec![
            0, 0, /**/ 1, 1, /**/ 2, 2, /**/ 3, 3, /**/ 0, 0
        ]
    );
    assert_eq!(
        (0..10)
            .map(|i| BidiSchedule::real_idx(&SymmetricStretched, i, 10, 4))
            .collect::<Vec<_>>(),
        vec![
            0, 0, /**/ 0, 0, /**/ 1, 1, /**/ 2, 2, /**/ 3, 3
        ]
    );
    let custom = vec![
        0, 1, /**/ 2, 2, /**/ 1, 0, /**/ 0, 0, /**/ 3, 2,
    ];
    assert_eq!(
        (0..10)
            .map(|i| BidiSchedule::real_idx(&Custom(custom.clone()), i, 10, 4))
            .collect::<Vec<_>>(),
        custom
    );
}

use super::*;

#[test]
fn test_cosine_annealing_warmup() {
    let scheduler = CosineAnnealingLr::new(1000)
        .with_max_lr(1.)
        .with_min_lr(0.01)
        .with_warmup_steps(100);

    // At step 0, LR should be 0.0001
    assert_eq!(scheduler.get_lr(0), 0.);

    // At half warmup, LR should be max_lr / 2
    assert_eq!(scheduler.get_lr(50), 0.5);

    // At end of warmup, LR should be max_lr
    assert_eq!(scheduler.get_lr(100), 1.0);

    // After total steps, LR should be min_lr
    assert_eq!(scheduler.get_lr(1000), 0.01);
}

#[test]
fn test_constant_lr() {
    let scheduler = ConstantLr::new().with_lr(0.001);
    assert_eq!(scheduler.get_lr(0), 0.001);
    assert_eq!(scheduler.get_lr(1000), 0.001);
    assert_eq!(scheduler.get_lr(10000), 0.001);
}

# 01: The untied tensors of the Kalman gain and of the tropical register in the block contracts

Status: resolved

Blocked by: none (can start immediately)

**What to build:** a Mamba-3 block can untie the tensors of its Kalman gain
(`Mamba3Untied::KalmanKappa`, `Mamba3Untied::KalmanRead`) and the readout of its tropical
register (`Mamba3Untied::TropicalReadout`). It can do this only when its config chooses a
Kalman `Gain` and `Tropical::MaxPlus`. The config, the init and `Mamba3::untied_params`
support this now, but no test reaches it. After this ticket:

- the block contracts check these untied tensors,
- a test checks that each of these untie choices panics when the config does not choose
  its tensor.

**The untied setting.** The block contracts have one untied Mamba-3 setting, with
`mimo_rank` 2, the output norm and the `Minimal` path. The user chose to extend this
setting, not to add a second one:

- It gets `Gain::KalmanProjectedNoise` and `Tropical::MaxPlus`. `KalmanProjectedNoise`
  has the two tensors of `Gain::Kalman`, and also the noise channel `r` of the
  in-projection.
- It also unties `KalmanKappa`, `KalmanRead` and `TropicalReadout`. So it unties each
  tensor of `Mamba3Untied` except `InitState`. Only the `Minimal` path reads the learnable
  initial state, and the step does not read it.
- It unties `InProjTail` already. So the untied tail then holds the noise channel and the
  two tropical channels, and the Muon contract checks the tiled spec of that tail.
- It keeps its path, `mimo_rank` 2 and the output norm.
- The header of the contract tests names what each untied setting unties. Its line for
  Mamba-3 names the tensors that the setting does not untie. Update that line.

**The panic test.** A rule-3 test beside `mamba3_layout_follows_the_untie_list`. Each of
these configs panics at init, with the message of its config check:

- `KalmanKappa` untied, with `Gain::Projected`,
- `KalmanRead` untied, with `Gain::Projected`,
- `TropicalReadout` untied, with `Tropical::None`.

A failure names the untie choice.

**Red check.** Each part has a temporary break in the family code, reverted after the
run:

- `Mamba3::untied_params` leaves out one of the three tensors. Then
  `untied_stack_equals_the_stack_of_its_application_views` fails on the untied Mamba-3
  setting, and the failure names that setting.
- One of the three config checks is gone. Then its panic test fails.

**When a contract fails.** A contract that fails on the extended setting can show a real
error of the Kalman gain or of the tropical register under untied copies. Stop and ask
the user. Do not remove a tensor from the setting to make it pass.

- [x] The untied Mamba-3 setting has `Gain::KalmanProjectedNoise` and `Tropical::MaxPlus`,
      and it unties `KalmanKappa`, `KalmanRead` and `TropicalReadout`.
- [x] Each block contract passes on that setting, on `flex`.
- [x] A test checks the panic of each of the three untie choices without its tensor.
- [x] Each break of the red check fails its test, and each break is reverted.
- [x] The header of the contract tests names what the untied Mamba-3 setting unties.
- [x] `cargo test --lib -- --test-threads=1` passes on `flex`. `scripts/check.sh` passes.

## Comments

### Where this comes from

- Ticket 06 of `.scratch/container-tests-home/` added the untied settings to the block
  contracts. Their source was the deleted test
  `mamba3_untied_stack_equals_the_stack_of_its_application_views`, which did not untie
  these three tensors. The review of that ticket found the gap.
- The untied tail of the in-projection starts at `dd_dt`. With
  `Gain::KalmanProjectedNoise` and `Tropical::MaxPlus`, it also holds the groups
  `KalmanNoise`, `TropicalA` and `TropicalB` of the layout table.

### The implementation

- `src/family/tests/block_contract.rs`: the untied Mamba-3 setting has the gain, the
  register and the three untie choices. Its name also gives the gain and the register.
  Each block contract passes on it, on `flex`. The family code did not change.
- `src/family/tests/untied.rs`: the test `mamba3_untie_choice_without_its_tensor_panics`.
  It compares the full text of each panic with the message of its config check.

### The red check

One run with two temporary breaks in `src/mamba3/mamba3.rs`, reverted after the run:

| Break | Test that fails | Line of the failure |
|---|---|---|
| `Mamba3::untied_params` gives no parameter for `KalmanKappa` | `untied_stack_equals_the_stack_of_its_application_views` | `Mamba-3, untied, …, KalmanProjectedNoise, MaxPlus`: the parameter `kalman_log_kappa_h` is not named |
| The config check of `TropicalReadout` is gone | `mamba3_untie_choice_without_its_tensor_panics` | `TropicalReadout: the init does not panic` |

The untied contract gave one line, for the untied Mamba-3 setting, and no line for the
other settings.

### The review

Two reports: `tmp/untied-kalman-tropical/review-standards.md` and
`tmp/untied-kalman-tropical/review-spec.md`. The fixes:

- The headers of `untied.rs` and of `src/family/tests/mod.rs` name the checks of the
  Kalman gain and of the tropical register. The test does not check the other config
  checks of the untie list.
- The header of `block_contract.rs` writes "channel group" (`GLOSSARY.md`).
- A failure of the panic test gives the panic text, not an `Option`.

Not changed:

- The panic test compares the full text of each message, as this ticket asks ("with the
  message of its config check"). A change of a message then also changes the test.
- `panic_message` of `block_contract.rs` repeats `panic_text` of burn-stack. It is
  outside this ticket.
- The untied setting has no double-SSD twin. So no contract runs the Kalman gain and
  the tropical register on the double-SSD pathway with untied copies.
- The Muon contract checks the width of the tiled spec of the tail, not the order of its
  groups.

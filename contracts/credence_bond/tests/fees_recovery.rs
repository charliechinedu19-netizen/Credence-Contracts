// Adversarial and failure-recovery coverage for `fees.rs` (issue #1327).
//
// Companion to `fees_boundaries.rs`, which covers values and shapes. This file
// covers what happens *around* a fee operation: retries after a failure, a
// sequence that fails partway, permission handover, emergency gates, and
// interleaving with the two other writers of the same balance.
//
// ## The three writers of one balance
//
// `fees::record_fee`, `CredenceBond::deposit_fees` and
// `CredenceBond::collect_fees` all read and write the raw key
// `Symbol::new(e, "fees")`. There is no `DataKey` variant for it. So the
// "fee pool" is a single accumulator with three writers and two different
// overflow behaviours:
//
// * `record_fee` — checked via `math::add_i128`, panics "fee pool overflow".
// * `deposit_fees` — raw `current + amount`, which wraps in a release/WASM
//   build.
// * `collect_fees` — resets to zero after reading.
//
// Every recovery case here that touches the pool is written against that
// shared-key reality, not against `record_fee` in isolation.
//
// ## Harnesses
//
// Entrypoint guards go through the generated client and assert the `Result`
// from `try_*`. Module helpers (`record_fee`, `set_config`, `get_config`) are
// plain library functions, so they are driven inside `env.as_contract`.
//
// ## Two host constraints that shape the file
//
// * `Env::events()` is frame-scoped: it only reports events published in the
//   current frame, so "no event emitted" is asserted as an empty log rather
//   than as a delta against a prior count.
// * A contract panic poisons the frame (`Contract re-entry is not allowed` on
//   the next call), so post-panic state is never asserted on the same `Env`.
//   Each panic test owns its `Env` and nothing else runs in it afterwards.

#![cfg(test)]

use credence_bond::fees::{get_config, record_fee, set_config, MAX_FEE_BPS};
use credence_bond::soroban_sdk::testutils::{Address as _, Events as _, Ledger as _};
use credence_bond::soroban_sdk::{Address, Bytes, Env, Symbol, TryIntoVal};
use credence_bond::{CredenceBond, CredenceBondClient};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::string::{String as StdString, ToString};

/// `ContractError::NotAdmin` — the host renders `panic_with_error!` as
/// `Error(Contract, #100)`.
const NOT_ADMIN_ERROR_CODE: u32 = credence_errors::ContractError::NotAdmin as u32;
/// `ContractError::ContractPaused` → `Error(Contract, #106)`.
const CONTRACT_PAUSED_ERROR_CODE: u32 = credence_errors::ContractError::ContractPaused as u32;
/// `ContractError::AmountMustBePositive`, the error `deposit_fees`' positive-
/// amount guard raises (`Error(Contract, #600)`).
const AMOUNT_MUST_BE_POSITIVE_ERROR_CODE: u32 =
    credence_errors::ContractError::AmountMustBePositive as u32;

fn deploy(e: &Env) -> (Address, CredenceBondClient<'_>) {
    let contract_id = e.register(CredenceBond, ());
    let client = CredenceBondClient::new(e, &contract_id);
    (contract_id, client)
}

/// Register, initialize, and authorize every address.
fn setup(e: &Env) -> (Address, CredenceBondClient<'_>, Address) {
    let (id, client) = deploy(e);
    let admin = Address::generate(e);
    e.mock_all_auths();
    client.initialize(&admin, &None);
    (id, client, admin)
}

fn bytes_of_len(e: &Env, len: usize) -> Bytes {
    // `std::vec!` repeat form; `soroban_sdk::vec!` is the element-list macro.
    Bytes::from_slice(e, &std::vec![7_u8; len])
}

/// The shared fee-pool key. There is no `DataKey` variant for it.
fn fee_pool_key(e: &Env) -> Symbol {
    Symbol::new(e, "fees")
}

/// Read the pool from inside a contract frame; instance storage is not
/// reachable from the test host.
fn read_fee_pool(e: &Env, contract_id: &Address) -> i128 {
    let mut out = -1_i128;
    e.as_contract(contract_id, || {
        out = e
            .storage()
            .instance()
            .get(&fee_pool_key(e))
            .unwrap_or(0_i128);
    });
    out
}

/// Write the pool directly, to set up a near-overflow baseline.
fn seed_fee_pool(e: &Env, contract_id: &Address, value: i128) {
    let key = fee_pool_key(e);
    e.as_contract(contract_id, || {
        e.storage().instance().set(&key, &value);
    });
}

/// The only observable form of the pool balance: `collect_fees` returns it and
/// resets it to zero.
fn drain_fee_pool(e: &Env, client: &CredenceBondClient<'_>, admin: &Address) -> i128 {
    client.collect_fees(admin, &bytes_of_len(e, 4))
}

/// Strip the host's panic envelope so the contract's own message can be pinned.
fn unwrap_contract_panic(raw: &str) -> StdString {
    const MARKER: &str = "caught panic '";
    let Some(start) = raw.find(MARKER) else {
        return raw.to_string();
    };
    let rest = &raw[start + MARKER.len()..];
    let Some(end) = rest.find('\'') else {
        return raw.to_string();
    };
    rest[..end].to_string()
}

fn panic_message<F: FnOnce()>(f: F) -> StdString {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    let raw = if let Some(s) = payload.downcast_ref::<StdString>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        panic!("panic payload was neither String nor &str");
    };
    unwrap_contract_panic(&raw)
}

#[track_caller]
fn expect_panic_with<F: FnOnce()>(expected: &str, f: F) {
    assert_eq!(panic_message(f), expected);
}

// ===========================================================================
// R1: Retry convergence — N failures must not drift state
// ===========================================================================

/// Retrying a rejected bounds write any number of times must leave the live
/// config untouched, and a corrected retry must then land. A setter that
/// partially applied on the failing path would show drift here that grows with
/// the retry count.
#[test]
fn repeated_bounds_rejection_never_drifts_the_live_config() {
    let e = Env::default();
    let (_id, client, admin) = setup(&e);
    let treasury = Address::generate(&e);
    let attacker = Address::generate(&e);

    client.set_fee_config(&admin, &treasury, &250);

    for attempt in 1..=8_u32 {
        assert!(
            client
                .try_set_fee_config(&admin, &attacker, &(MAX_FEE_BPS + attempt))
                .is_err(),
            "attempt {attempt} must be rejected"
        );
        assert_eq!(
            client.get_fee_config(),
            (Some(treasury.clone()), 250),
            "attempt {attempt} drifted the stored config"
        );
    }

    // The corrected retry converges on the first try.
    client.set_fee_config(&admin, &treasury, &300);
    assert_eq!(client.get_fee_config(), (Some(treasury), 300));
}

/// Same convergence for the permission guard: probing with different stranger
/// addresses must not move the config or the admin.
#[test]
fn repeated_unauthorised_probes_never_drift_the_live_config() {
    let e = Env::default();
    let (_id, client, admin) = setup(&e);
    let treasury = Address::generate(&e);

    client.set_fee_config(&admin, &treasury, &250);

    for _ in 0..8 {
        let stranger = Address::generate(&e);
        assert!(client
            .try_set_fee_config(&stranger, &stranger, &999)
            .is_err());
        assert_eq!(client.get_fee_config(), (Some(treasury.clone()), 250));
    }

    // The real admin still works afterwards: the probes did not latch anything.
    client.set_fee_config(&admin, &treasury, &400);
    assert_eq!(client.get_fee_config(), (Some(treasury), 400));
}

/// Every rejection path emits nothing. An indexer replaying the log must see a
/// clean sequence of successful writes only — no interleaved rejected values.
#[test]
fn no_rejection_path_emits_an_event() {
    let e = Env::default();
    let (_id, client, admin) = setup(&e);
    let treasury = Address::generate(&e);
    let stranger = Address::generate(&e);

    // Each of these is a distinct rejection: bounds, then permission.
    assert!(client
        .try_set_fee_config(&admin, &treasury, &(MAX_FEE_BPS + 1))
        .is_err());
    assert!(e.events().all().is_empty(), "bounds rejection emitted");

    assert!(client
        .try_set_fee_config(&stranger, &stranger, &100)
        .is_err());
    assert!(e.events().all().is_empty(), "permission rejection emitted");

    client.pause(&admin);
    assert!(client.try_set_fee_config(&admin, &treasury, &100).is_err());
    assert!(e.events().all().is_empty(), "paused rejection emitted");
}

/// A rejected write is invisible to the module getter as well as the
/// entrypoint, so there is no second path onto the half-written state.
#[test]
fn rejection_is_invisible_to_the_module_getter_too() {
    let e = Env::default();
    let (contract_id, client, admin) = setup(&e);
    let treasury = Address::generate(&e);
    let attacker = Address::generate(&e);

    client.set_fee_config(&admin, &treasury, &250);

    assert!(client
        .try_set_fee_config(&admin, &attacker, &(MAX_FEE_BPS + 1))
        .is_err());

    let mut module_view = (None, 0);
    e.as_contract(&contract_id, || {
        module_view = get_config(&e);
    });
    assert_eq!(module_view, (Some(treasury), 250));
}

// ===========================================================================
// R2: Partial failure across a sequence
// ===========================================================================

/// A sequence that fails partway keeps the writes that already committed.
/// Soroban reverts per invocation, not per batch, so governance must be able to
/// resume from wherever it stopped rather than restart from scratch.
#[test]
fn a_failing_write_in_a_sequence_leaves_prior_writes_intact() {
    let e = Env::default();
    let (_id, client, admin) = setup(&e);
    let treasury_a = Address::generate(&e);
    let treasury_b = Address::generate(&e);

    client.set_fee_config(&admin, &treasury_a, &100);
    // This one fails.
    assert!(client
        .try_set_fee_config(&admin, &treasury_b, &(MAX_FEE_BPS + 1))
        .is_err());
    // This one must still land.
    client.set_fee_config(&admin, &treasury_b, &200);

    assert_eq!(client.get_fee_config(), (Some(treasury_b), 200));
}

/// A rejection between two successful writes does not corrupt the event
/// sequence: the successful write after it still reports the *last successful*
/// values as its old values, not the rejected ones. This is the property that
/// keeps a replaying indexer's view aligned with storage.
#[test]
fn a_rejection_between_writes_does_not_corrupt_the_audit_trail() {
    let e = Env::default();
    let (contract_id, client, admin) = setup(&e);
    let treasury_a = Address::generate(&e);
    let treasury_b = Address::generate(&e);
    let rejected_treasury = Address::generate(&e);

    // Successful write, recorded from inside the frame that made it.
    let mut after_first = None;
    e.as_contract(&contract_id, || {
        set_config(&e, &admin, treasury_a.clone(), 100);
        let (_contract, topics, data) = e.events().all().last().expect("event");
        let raw: Vec<credence_bond::soroban_sdk::Val> = topics.iter().collect();
        assert_eq!(raw.len(), 2);
        let payload: (Option<Address>, Address, u32, u32) = data
            .clone()
            .try_into_val(&e)
            .expect("payload is the 4-tuple");
        after_first = Some(payload);
    });
    let (old_t, new_t, old_bps, new_bps) = after_first.expect("payload");
    assert_eq!(old_t, None);
    assert_eq!(new_t, treasury_a.clone());
    assert_eq!(old_bps, 0);
    assert_eq!(new_bps, 100);

    // Rejected write in between, on its own frame.
    assert!(client
        .try_set_fee_config(&admin, &rejected_treasury, &(MAX_FEE_BPS + 1))
        .is_err());

    // The next successful write must report treasury_a / 100 as its old values.
    let mut after_third = None;
    e.as_contract(&contract_id, || {
        set_config(&e, &admin, treasury_b.clone(), 200);
        let (_contract, _topics, data) = e.events().all().last().expect("event");
        let payload: (Option<Address>, Address, u32, u32) = data
            .clone()
            .try_into_val(&e)
            .expect("payload is the 4-tuple");
        after_third = Some(payload);
    });
    let (old_t, new_t, old_bps, new_bps) = after_third.expect("payload");
    assert_eq!(
        old_t,
        Some(treasury_a),
        "old value must be the last successful write, not the rejected one"
    );
    assert_eq!(new_t, treasury_b);
    assert_eq!(old_bps, 100, "old rate must skip the rejected write");
    assert_eq!(new_bps, 200);
}

/// Interleaved writes across the two independent fee surfaces — the config
/// (treasury + rate) and the pool (accrued balance) — must not interfere. They
/// are different keys, and a fee-config governance action must never disturb
/// money already accrued.
#[test]
fn config_writes_and_pool_writes_are_independent() {
    let e = Env::default();
    let (contract_id, client, admin) = setup(&e);
    let treasury = Address::generate(&e);

    client.set_fee_config(&admin, &treasury, &250);
    client.deposit_fees(&5_000);

    // A rejected config write between two pool operations.
    assert!(client
        .try_set_fee_config(&admin, &Address::generate(&e), &(MAX_FEE_BPS + 1))
        .is_err());
    client.deposit_fees(&1_000);

    // A pool sweep between two config writes.
    assert_eq!(drain_fee_pool(&e, &client, &admin), 6_000);
    client.set_fee_config(&admin, &treasury, &750);

    assert_eq!(client.get_fee_config(), (Some(treasury), 750));
    assert_eq!(
        read_fee_pool(&e, &contract_id),
        0,
        "a sweep must leave the pool at zero"
    );
}

// ===========================================================================
// R3: Permission handover
// ===========================================================================

/// After `transfer_admin`, the old admin is locked out of the fee config and
/// the new one works, with no drift in the configured rate.
#[test]
fn admin_handover_moves_fee_config_authority_without_drift() {
    let e = Env::default();
    let (_id, client, admin) = setup(&e);
    let successor = Address::generate(&e);
    let treasury = Address::generate(&e);

    client.set_fee_config(&admin, &treasury, &250);
    client.transfer_admin(&admin, &successor);

    // Old admin is out.
    assert!(client
        .try_set_fee_config(&admin, &Address::generate(&e), &900)
        .is_err());
    assert_eq!(client.get_fee_config(), (Some(treasury.clone()), 250));

    // New admin is in.
    client.set_fee_config(&successor, &treasury, &600);
    assert_eq!(client.get_fee_config(), (Some(treasury), 600));
}

/// Handover also moves the authority to *drain* the pool. A rejected sweep by
/// the old admin must leave the balance intact for the new one — otherwise the
/// handover would let a stale admin probe the balance down to zero.
#[test]
fn admin_handover_moves_fee_pool_drain_authority() {
    let e = Env::default();
    let (_id, client, admin) = setup(&e);
    let successor = Address::generate(&e);

    client.deposit_fees(&4_200);
    client.transfer_admin(&admin, &successor);

    assert!(client
        .try_collect_fees(&admin, &bytes_of_len(&e, 4))
        .is_err());

    assert_eq!(drain_fee_pool(&e, &client, &successor), 4_200);
}

/// The wire code for a stale admin is pinned so the handover cannot silently
/// start failing with a different error.
#[test]
fn stale_admin_after_handover_fails_with_not_admin() {
    let e = Env::default();
    let (_id, client, admin) = setup(&e);
    let successor = Address::generate(&e);

    client.transfer_admin(&admin, &successor);

    let err = client
        .try_set_fee_config(&admin, &Address::generate(&e), &100)
        .expect_err("stale admin must not set the fee config");
    let rendered = format!("{err:?}");
    assert!(
        rendered.contains(&format!("#{NOT_ADMIN_ERROR_CODE}")),
        "expected NotAdmin #{NOT_ADMIN_ERROR_CODE}, got {rendered}"
    );
}

// ===========================================================================
// R4: Emergency gates
// ===========================================================================

/// Pause blocks the fee-config write, and the rejection is `ContractPaused`, so
/// a keeper can distinguish "paused" from "out of bounds" and "not admin".
#[test]
fn pause_blocks_config_writes_with_the_paused_error() {
    let e = Env::default();
    let (_id, client, admin) = setup(&e);
    let treasury = Address::generate(&e);

    client.pause(&admin);

    let err = client
        .try_set_fee_config(&admin, &treasury, &500)
        .expect_err("paused contract must reject set_fee_config");
    let rendered = format!("{err:?}");
    assert!(
        rendered.contains(&format!("#{CONTRACT_PAUSED_ERROR_CODE}")),
        "expected ContractPaused #{CONTRACT_PAUSED_ERROR_CODE}, got {rendered}"
    );
}

/// Pause is not a data-loss event: a blocked write during the pause leaves the
/// previous config intact, and unpausing lets the same call land.
#[test]
fn pause_then_unpause_converges_with_no_drift() {
    let e = Env::default();
    let (_id, client, admin) = setup(&e);
    let treasury = Address::generate(&e);

    client.set_fee_config(&admin, &treasury, &100);

    client.pause(&admin);
    for rate in [200_u32, 300, 400, 500] {
        assert!(client
            .try_set_fee_config(&admin, &Address::generate(&e), &rate)
            .is_err());
        assert_eq!(client.get_fee_config(), (Some(treasury.clone()), 100));
    }

    client.unpause(&admin);
    client.set_fee_config(&admin, &treasury, &500);
    assert_eq!(client.get_fee_config(), (Some(treasury), 500));
}

/// Pause also blocks deposits and sweeps, so an emergency stop freezes every
/// writer of the shared balance — not just the governance path.
#[test]
fn pause_blocks_every_writer_of_the_shared_balance() {
    let e = Env::default();
    let (_id, client, admin) = setup(&e);

    client.deposit_fees(&1_000);
    client.pause(&admin);

    assert!(client.try_deposit_fees(&500).is_err());
    assert!(client
        .try_collect_fees(&admin, &bytes_of_len(&e, 4))
        .is_err());

    client.unpause(&admin);
    // Nothing was lost or double-counted by the blocked attempts.
    assert_eq!(drain_fee_pool(&e, &client, &admin), 1_000);
}

// ===========================================================================
// R5: The shared balance — overflow, wrap, and sweep
// ===========================================================================

/// `record_fee` is checked: a fee that would push the pool past `i128::MAX`
/// panics `"fee pool overflow"` rather than wrapping. The pool is left at its
/// prior value, which is what makes the failure recoverable — the caller can
/// sweep and retry at a smaller fee.
#[test]
fn record_fee_panics_on_overflow_and_leaves_the_prior_balance() {
    let e = Env::default();
    let (contract_id, _client, _admin) = setup(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    // A baseline well below the ceiling, so "prior value" is a real number.
    seed_fee_pool(&e, &contract_id, 1_000);

    expect_panic_with("fee pool overflow", || {
        e.as_contract(&contract_id, || {
            record_fee(&e, &identity, 1, i128::MAX, &treasury);
        });
    });

    // The panic poisoned the frame, so the balance is checked on a fresh read.
    assert_eq!(read_fee_pool(&e, &contract_id), 1_000);
}

/// The boundary: a fee that lands exactly on `i128::MAX` is accepted. This is
/// the last accepted value, and it separates the overflow case above from an
/// off-by-one in the checked addition.
#[test]
fn record_fee_accepts_a_fee_landing_exactly_on_i128_max() {
    let e = Env::default();
    let (contract_id, _client, _admin) = setup(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    seed_fee_pool(&e, &contract_id, 0);
    e.as_contract(&contract_id, || {
        record_fee(&e, &identity, 1, i128::MAX, &treasury);
    });

    assert_eq!(read_fee_pool(&e, &contract_id), i128::MAX);
}

/// One past the ceiling is rejected by the same guard.
#[test]
fn record_fee_rejects_one_past_i128_max() {
    let e = Env::default();
    let (contract_id, _client, _admin) = setup(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    seed_fee_pool(&e, &contract_id, 1);
    expect_panic_with("fee pool overflow", || {
        e.as_contract(&contract_id, || {
            record_fee(&e, &identity, 1, i128::MAX, &treasury);
        });
    });
    assert_eq!(read_fee_pool(&e, &contract_id), 1);
}

/// Recovery from a saturated pool: drain it, then the accumulator works again.
/// Without this the contract would be permanently unable to record a fee once
/// it ever reached the ceiling.
///
/// `collect_fees` reads the balance and resets it without adding, so it is
/// unaffected by the saturation that blocks `record_fee`.
#[test]
fn draining_a_saturated_pool_restores_the_accumulator() {
    let e = Env::default();
    let (contract_id, client, admin) = setup(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    seed_fee_pool(&e, &contract_id, i128::MAX);

    // `collect_fees` reads and resets without adding, so it still works.
    assert_eq!(drain_fee_pool(&e, &client, &admin), i128::MAX);
    assert_eq!(read_fee_pool(&e, &contract_id), 0);

    // And the accumulator is usable again.
    e.as_contract(&contract_id, || {
        record_fee(&e, &identity, 1_000_000, 250, &treasury);
    });
    assert_eq!(read_fee_pool(&e, &contract_id), 250);
    assert_eq!(drain_fee_pool(&e, &client, &admin), 250);
}

/// A saturated pool does block accrual — `record_fee` cannot add anything to a
/// balance already at the ceiling. This is the failure the drain above exists
/// to clear, and it is asserted on its own `Env` because the panic poisons the
/// frame.
#[test]
fn a_saturated_pool_blocks_accrual_until_drained() {
    let e = Env::default();
    let (contract_id, _client, _admin) = setup(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    seed_fee_pool(&e, &contract_id, i128::MAX);

    expect_panic_with("fee pool overflow", || {
        e.as_contract(&contract_id, || {
            record_fee(&e, &identity, 1, 1, &treasury);
        });
    });
}

/// The two writers of the shared balance do **not** share an overflow
/// strategy. `record_fee` adds through `math::add_i128`, which raises the
/// contract's own `"fee pool overflow"` diagnostic regardless of build
/// profile. `deposit_fees` adds with a bare `current + amount`, so the
/// overflow is the language's, not the contract's — and therefore
/// build-dependent: under the test profile's overflow checks it is an
/// `"attempt to add with overflow"` trap, while a release/WASM build has
/// overflow checks off and wraps silently to a negative balance.
///
/// Only the part that holds in every profile is asserted here: the two
/// surfaces do not produce the same diagnostic, so they are not
/// interchangeable and an operator cannot rely on one message meaning the
/// other. The release-build wrap is not reachable from a test.
#[test]
fn deposit_fees_does_not_use_the_checked_add_helper() {
    // Same input, two surfaces, two independent `Env`s: a panic poisons its
    // frame, so the two cases cannot share one.
    let input_floor = i128::MAX - 10;
    let addition = 100_i128;

    let e = Env::default();
    let (contract_id, _client, _admin) = setup(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);
    seed_fee_pool(&e, &contract_id, input_floor);
    let record_msg = panic_message(|| {
        e.as_contract(&contract_id, || {
            record_fee(&e, &identity, 1, addition, &treasury);
        });
    });

    let e2 = Env::default();
    let (id2, client2, _admin2) = setup(&e2);
    seed_fee_pool(&e2, &id2, input_floor);
    let deposit_msg = panic_message(|| {
        client2.deposit_fees(&addition);
    });

    assert_eq!(
        record_msg, "fee pool overflow",
        "record_fee must raise the contract's own diagnostic"
    );
    assert_ne!(
        deposit_msg, "fee pool overflow",
        "deposit_fees bypassed math::add_i128, so it cannot raise that message; \
         got {deposit_msg:?}"
    );
    assert_eq!(
        deposit_msg, "attempt to add with overflow",
        "under the test profile the bare `+` traps with the language diagnostic; \
         a release build wraps silently instead, which no test can observe"
    );
}

/// The overflow leaves the pool untouched under `record_fee`, so the balance is
/// still exactly what it was before the attempt. This is what makes the
/// saturation recoverable: nothing is half-written.
#[test]
fn a_failed_record_fee_leaves_the_pool_byte_for_byte_intact() {
    let e = Env::default();
    let (contract_id, _client, _admin) = setup(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    // Close enough to the ceiling that adding a whole unit of fee overflows.
    let before = i128::MAX - 1;
    seed_fee_pool(&e, &contract_id, before);

    expect_panic_with("fee pool overflow", || {
        e.as_contract(&contract_id, || {
            record_fee(&e, &identity, 1, 1_000, &treasury);
        });
    });

    assert_eq!(read_fee_pool(&e, &contract_id), before);
}

/// `deposit_fees` rejects a non-positive *argument* even though `record_fee`
/// silently no-ops on the same values. A caller that retries with zero to probe
/// the pool state gets an error from one surface and silence from the other.
#[test]
fn deposit_fees_rejects_the_values_record_fee_silently_ignores() {
    let e = Env::default();
    let (contract_id, client, _admin) = setup(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    // `record_fee` no-ops, leaving no trace.
    e.as_contract(&contract_id, || {
        record_fee(&e, &identity, 1_000, 0, &treasury);
        record_fee(&e, &identity, 1_000, -50, &treasury);
    });
    assert_eq!(read_fee_pool(&e, &contract_id), 0);

    // `deposit_fees` errors on the same values, with the positive-amount code.
    for bad in [0_i128, -1] {
        let err = client
            .try_deposit_fees(&bad)
            .expect_err("deposit_fees must reject a non-positive amount");
        let rendered = format!("{err:?}");
        assert!(
            rendered.contains(&format!("#{AMOUNT_MUST_BE_POSITIVE_ERROR_CODE}")),
            "deposit_fees({bad}): expected InvalidAmount              #{AMOUNT_MUST_BE_POSITIVE_ERROR_CODE}, got {rendered}"
        );
    }
}

/// `collect_fees` resets the balance to zero, so a second sweep returns zero
/// rather than repeating the payout. This is the property that stops an
/// accidental double-sweep from double-paying the treasury.
#[test]
fn a_second_sweep_returns_zero_not_the_previous_balance() {
    let e = Env::default();
    let (_id, client, admin) = setup(&e);

    client.deposit_fees(&3_333);
    assert_eq!(drain_fee_pool(&e, &client, &admin), 3_333);
    assert_eq!(drain_fee_pool(&e, &client, &admin), 0);
    assert_eq!(drain_fee_pool(&e, &client, &admin), 0);
}

/// The balance survives a sweep cycle: whatever accrues after a sweep starts
/// from zero and is paid out in full on the next sweep.
#[test]
fn accrual_after_a_sweep_starts_from_zero() {
    let e = Env::default();
    let (contract_id, client, admin) = setup(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    client.deposit_fees(&1_000);
    assert_eq!(drain_fee_pool(&e, &client, &admin), 1_000);

    // A mix of the two dormant and one live writer.
    e.as_contract(&contract_id, || {
        record_fee(&e, &identity, 500_000, 750, &treasury);
        record_fee(&e, &identity, 500_000, 250, &treasury);
    });
    client.deposit_fees(&99);

    assert_eq!(read_fee_pool(&e, &contract_id), 1_099);
    assert_eq!(drain_fee_pool(&e, &client, &admin), 1_099);
}

/// `record_fee` writes the same key the contract entrypoints read, so a sweep
/// pays out dormant accruals too. Proving the shared key is what lets the rest
/// of this file treat the three writers as one balance.
#[test]
fn a_sweep_pays_out_dormant_record_fee_accruals() {
    let e = Env::default();
    let (contract_id, client, admin) = setup(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    e.as_contract(&contract_id, || {
        record_fee(&e, &identity, 1_000_000, 900, &treasury);
    });

    // Reachable through the live entrypoint with no dormant code involved.
    assert_eq!(drain_fee_pool(&e, &client, &admin), 900);
}

/// A failed sweep must not reset the balance. `collect_fees` zeroes the key
/// before the callback, so a re-entrant or otherwise failing sweep is the case
/// where that ordering matters; here the simpler guarantee is pinned — an
/// unauthorised sweep leaves the money where it was.
#[test]
fn a_rejected_sweep_leaves_the_shared_balance_intact() {
    let e = Env::default();
    let (contract_id, client, admin) = setup(&e);
    let stranger = Address::generate(&e);

    client.deposit_fees(&2_500);
    for _ in 0..4 {
        assert!(client
            .try_collect_fees(&stranger, &bytes_of_len(&e, 4))
            .is_err());
        assert_eq!(read_fee_pool(&e, &contract_id), 2_500);
    }

    assert_eq!(drain_fee_pool(&e, &client, &admin), 2_500);
}

/// The reentrancy lock is not held across a fee-config write: `set_fee_config`
/// deliberately omits `acquire_lock` because it makes no cross-contract call.
/// Pinned so that omission stays a reviewed decision rather than an oversight —
/// and so the test would fail if a lock were added without a matching release.
#[test]
fn set_fee_config_does_not_take_the_reentrancy_lock() {
    let e = Env::default();
    let (_id, client, admin) = setup(&e);
    let treasury = Address::generate(&e);

    client.set_fee_config(&admin, &treasury, &250);
    assert!(
        !client.is_locked(),
        "set_fee_config must not leave the reentrancy lock held"
    );

    // And a later lock-taking operation still works, which it would not if the
    // previous call had leaked the lock.
    client.deposit_fees(&100);
    assert_eq!(drain_fee_pool(&e, &client, &admin), 100);
    assert!(!client.is_locked());
}

/// `collect_fees` takes and releases the lock around the whole sweep, so the
/// balance is restored to an unlocked state afterwards. A leaked lock here would
/// wedge every other guarded entrypoint permanently.
#[test]
fn collect_fees_releases_the_reentrancy_lock() {
    let e = Env::default();
    let (_id, client, admin) = setup(&e);

    client.deposit_fees(&500);
    assert_eq!(drain_fee_pool(&e, &client, &admin), 500);
    assert!(
        !client.is_locked(),
        "collect_fees must release the lock even on the success path"
    );

    // A second guarded call succeeding is the observable proof.
    client.deposit_fees(&1);
    assert_eq!(drain_fee_pool(&e, &client, &admin), 1);
}

// ===========================================================================
// R6: Ledger-time independence
// ===========================================================================

/// The fee config has no time component: advancing the ledger arbitrarily far
/// must not expire, decay, or reset it. A config that silently reverted after a
/// TTL bump would let a fee disappear without any governance action.
#[test]
fn config_survives_an_arbitrarily_long_ledger_advance() {
    let e = Env::default();
    let (_id, client, admin) = setup(&e);
    let treasury = Address::generate(&e);

    client.set_fee_config(&admin, &treasury, &250);

    for seconds in [1_u64, 86_400, 30 * 86_400, 365 * 86_400] {
        e.ledger().with_mut(|li| li.timestamp += seconds);
        assert_eq!(
            client.get_fee_config(),
            (Some(treasury.clone()), 250),
            "config drifted after advancing {seconds}s"
        );
    }
}

/// The same holds for the accrued balance: time alone does not sweep or decay
/// the pool.
#[test]
fn the_shared_balance_survives_a_ledger_advance() {
    let e = Env::default();
    let (contract_id, client, admin) = setup(&e);

    client.deposit_fees(&777);

    e.ledger().with_mut(|li| li.timestamp += 400 * 86_400_u64);

    assert_eq!(read_fee_pool(&e, &contract_id), 777);
    assert_eq!(drain_fee_pool(&e, &client, &admin), 777);
}

/// An expired stale config is not a concept here: writing the same config twice
/// separated by a long gap still emits both times, so an indexer sees the
/// governance timeline rather than a silent no-op.
#[test]
fn repeated_identical_writes_emit_each_time_across_time() {
    let e = Env::default();
    let (contract_id, _client, admin) = setup(&e);
    let treasury = Address::generate(&e);

    // The first write reports the unconfigured prior state; every round after
    // it must report the identical old/new pair, so a long gap in the ledger
    // never turns a re-set into a silent no-op.
    let mut prior: Option<(Option<Address>, Address, u32, u32)> = None;

    for round in 1..=3_u32 {
        e.ledger()
            .with_mut(|li| li.timestamp += round as u64 * 86_400);

        let mut seen = None;
        e.as_contract(&contract_id, || {
            set_config(&e, &admin, treasury.clone(), 250);
            let (_contract, _topics, data) = e.events().all().last().expect("event");
            let payload: (Option<Address>, Address, u32, u32) = data
                .clone()
                .try_into_val(&e)
                .expect("payload is the 4-tuple");
            seen = Some(payload);
        });

        let payload = seen.expect("payload present");
        match prior {
            None => assert_eq!(
                payload,
                (None, treasury.clone(), 0, 250),
                "round 1 must report the unconfigured prior state"
            ),
            Some(previous) => assert_eq!(
                payload,
                (Some(treasury.clone()), treasury.clone(), 250, 250),
                "round {round} must report identical old/new values; \
                 previous event was {previous:?}"
            ),
        }
        prior = Some(payload);
    }
}

// ===========================================================================
// R7: Dormancy — the invariants that hold because nothing is wired up
// ===========================================================================

/// `create_bond` stores `bonded_amount` verbatim and charges nothing, no matter
/// how the fee config is set. This is the single most important recovery fact
/// in the file: the fee surface cannot strand a bond, because it is not in the
/// bond's path at all.
///
/// If a future change wires fee collection into `create_bond`, this test fails
/// — which is the intended signal to revisit the whole dormant characterisation.
#[test]
fn bond_creation_is_unaffected_by_the_fee_config() {
    let e = Env::default();
    let (contract_id, client, admin) = setup(&e);
    let treasury = Address::generate(&e);

    let identity = Address::generate(&e);
    // `create_bond` rejects anything under `MIN_BOND_AMOUNT` (1e18), so the
    // amount has to be large enough to form a real bond.
    let amount = 10_i128.pow(18);

    // Maximum fee, live treasury.
    client.set_fee_config(&admin, &treasury, &MAX_FEE_BPS);

    let bond = client.create_bond(&identity, &amount, &10_000_u64, &false, &0_u64);

    assert_eq!(
        bond.bonded_amount, amount,
        "the bonded amount must be stored verbatim, with no fee deducted"
    );
    assert_eq!(
        read_fee_pool(&e, &contract_id),
        0,
        "bond creation must not accrue anything to the fee pool"
    );
    assert_eq!(
        drain_fee_pool(&e, &client, &admin),
        0,
        "nothing is collectable after bond creation"
    );
}

/// The same at a zero rate, so the assertion is not merely "a zero fee is
/// trivially zero".
#[test]
fn bond_creation_is_unaffected_at_every_configured_rate() {
    let e = Env::default();
    let (contract_id, client, admin) = setup(&e);
    let treasury = Address::generate(&e);

    for rate in [0_u32, 1, 100, 500, MAX_FEE_BPS] {
        client.set_fee_config(&admin, &treasury, &rate);

        let identity = Address::generate(&e);
        let amount = 10_i128.pow(18);
        let bond = client.create_bond(&identity, &amount, &10_000_u64, &false, &0_u64);

        assert_eq!(
            bond.bonded_amount, amount,
            "rate {rate} changed the bonded amount"
        );
        assert_eq!(
            read_fee_pool(&e, &contract_id),
            0,
            "rate {rate} accrued a fee"
        );
    }
}

/// Configure-then-charge cannot be observed through any entrypoint: the
/// configured rate is readable, and `create_bond` still charges nothing. The
/// two halves are asserted together so the dormancy claim is falsifiable from
/// a single test.
#[test]
fn the_configured_rate_is_readable_but_still_never_charged() {
    let e = Env::default();
    let (contract_id, client, admin) = setup(&e);
    let treasury = Address::generate(&e);

    client.set_fee_config(&admin, &treasury, &500);
    assert_eq!(
        client.get_fee_config(),
        (Some(treasury), 500),
        "the rate is live and auditable"
    );

    let identity = Address::generate(&e);
    client.create_bond(&identity, &10_i128.pow(18), &10_000_u64, &false, &0_u64);

    assert_eq!(
        read_fee_pool(&e, &contract_id),
        0,
        "a 5% configured rate still charges nothing at bond creation"
    );
}

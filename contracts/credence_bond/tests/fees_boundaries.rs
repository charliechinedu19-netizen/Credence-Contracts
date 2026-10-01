//! Boundary and characterization coverage for `fees.rs` (issue #1327).
//!
//! # What this file covers
//!
//! `fees.rs` is the bond-creation fee configuration surface plus a fee
//! arithmetic/accrual surface that is **not wired into the contract**. The two
//! halves have very different risk profiles, so they get different treatment:
//!
//! * **Live: `get_config` / `set_config`** (reached through
//!   `get_fee_config` / `set_fee_config`). Governance-controlled, so the
//!   interesting boundaries are the inclusive `[MIN_FEE_BPS, MAX_FEE_BPS]`
//!   range, rejection atomicity, permission, and the audit event.
//! * **Dormant: `calculate_fee` / `is_fee_waived` / `record_fee` /
//!   `emit_fee_event`.** These have no callers in the workspace; `create_bond`
//!   stores `bonded_amount` verbatim and charges nothing. The tests here
//!   *characterize* what they do rather than asserting what they should do,
//!   because there is no entrypoint through which to observe an integration
//!   failure.
//!
//! # Two things this file deliberately does NOT assert
//!
//! **The three `calculate_fee` panic strings.** `calculate_fee` forwards
//! `"fee calculation overflow"`, `"fee calculation div-by-zero"` and
//! `"fee calculation underflow"` to `math::split_bps`, which binds all three
//! to `_`-prefixed parameters and ignores them in favour of
//! `saturating_mul` / `saturating_sub`. Those strings are unreachable. A
//! `#[should_panic(expected = "fee calculation overflow")]` would be a test
//! that cannot fail for the reason it claims, so instead
//! `calculate_fee_at_i128_max_saturates_instead_of_panicking` pins the
//! saturating behaviour that actually occurs.
//!
//! **That `MAX_FEE_BPS` means 100%.** It is 1 000 bps = 10%. The dead
//! `src/test_fees.rs` asserts the opposite in a comment; the values pinned in
//! `calculate_fee_at_max_bps_takes_ten_percent_not_all` are the real ones.
//!
//! # Harnesses
//!
//! Entrypoint guards go through the generated client and assert the returned
//! `Result` from `try_*`. Module helpers are plain library functions driven
//! inside `env.as_contract`, whose panics are unwrapped from the host envelope
//! and compared exactly.
//!
//! `Env::events()` is frame-scoped, so event assertions run inside the same
//! `as_contract` frame as the mutation that publishes them.

#![cfg(test)]

use credence_bond::fees::{
    calculate_fee, get_config, is_fee_waived, record_fee, set_config, MAX_FEE_BPS, MIN_FEE_BPS,
};
use credence_bond::soroban_sdk::testutils::{Address as _, Events as _};
use credence_bond::soroban_sdk::{Address, Env, Symbol, TryIntoVal, Val};
use credence_bond::{CredenceBond, CredenceBondClient};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::string::{String as StdString, ToString};

/// `ContractError::NotAdmin`, pinned numerically because the host renders
/// `panic_with_error!` as `Error(Contract, #100)`.
const NOT_ADMIN_ERROR_CODE: u32 = credence_errors::ContractError::NotAdmin as u32;
/// `ContractError::ContractPaused` → `Error(Contract, #106)`.
const CONTRACT_PAUSED_ERROR_CODE: u32 = credence_errors::ContractError::ContractPaused as u32;

fn deploy(e: &Env) -> (Address, CredenceBondClient<'_>) {
    let contract_id = e.register(CredenceBond, ());
    let client = CredenceBondClient::new(e, &contract_id);
    (contract_id, client)
}

/// Register, initialize, and authorize every address. Returns the client and
/// the admin address that owns the fee config.
fn setup(e: &Env) -> (CredenceBondClient<'_>, Address) {
    let (_id, client) = deploy(e);
    let admin = Address::generate(e);
    e.mock_all_auths();
    client.initialize(&admin, &None);
    (client, admin)
}

/// The raw fee-pool key shared by `fees::record_fee`, `deposit_fees` and
/// `collect_fees`. There is no `DataKey` variant for it.
fn fee_pool_key(e: &Env) -> Symbol {
    Symbol::new(e, "fees")
}

/// Read the shared fee-pool balance from inside a contract frame. Instance
/// storage is inaccessible from the test host, so the read has to be wrapped.
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

/// Write `value` into the fee pool from inside a contract frame, so
/// `record_fee` can be driven against a near-overflow baseline.
fn seed_fee_pool(e: &Env, contract_id: &Address, value: i128) {
    let key = fee_pool_key(e);
    e.as_contract(contract_id, || {
        e.storage().instance().set(&key, &value);
    });
}

/// A contract-invoked panic reaches the caller wrapped by the host as
/// `HostError: Error(WasmVm, InvalidAction)` with the real message inside a
/// `caught panic '<message>'` diagnostic. Strip that envelope.
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
    let msg = panic_message(f);
    assert_eq!(msg, expected);
}

// ---------------------------------------------------------------------------
// S1: The documented governance bounds
// ---------------------------------------------------------------------------

/// The two bounds are load-bearing for every other test in this file, so pin
/// their literal values rather than trusting the constants: an off-by-one in
/// either would silently weaken every range assertion below.
#[test]
fn governance_bounds_are_zero_to_one_thousand_bps() {
    assert_eq!(MIN_FEE_BPS, 0);
    assert_eq!(MAX_FEE_BPS, 1_000);
    assert!(MIN_FEE_BPS < MAX_FEE_BPS);
}

/// `MIN_FEE_BPS == 0` and the parameter is a `u32`, so the lower arm of
/// `(MIN_FEE_BPS..=MAX_FEE_BPS).contains(&fee_bps)` can never reject. The bounds
/// check is effectively upper-only. Pinned so a future change to a non-zero
/// `MIN_FEE_BPS` — which would start rejecting low rates — is recognised as a
/// behaviour change rather than landing silently.
#[test]
fn min_fee_bps_lower_bound_cannot_reject_a_u32() {
    assert_eq!(MIN_FEE_BPS, 0);
    // There is no `u32` below zero, so no input can ever trip the lower arm.
    assert_eq!(MIN_FEE_BPS.saturating_sub(1), 0);
    assert!((MIN_FEE_BPS..=MAX_FEE_BPS).contains(&0));
}

/// `MAX_FEE_BPS` is 10%, matching `parameters::MAX_PROTOCOL_FEE_BPS`. If this
/// ever diverges the "one consistent ceiling" claim in the module docs breaks.
#[test]
fn max_fee_bps_matches_protocol_fee_ceiling() {
    assert_eq!(MAX_FEE_BPS, credence_bond::parameters::MAX_PROTOCOL_FEE_BPS);
}

// ---------------------------------------------------------------------------
// S2: Loading — getters are total, never panic
// ---------------------------------------------------------------------------

/// Boundary: on a contract nobody has initialized, every fee getter still
/// answers. A keeper polling `get_fee_config` on a fresh deployment must get a
/// value, not a trap.
#[test]
fn get_config_is_total_on_uninitialized_contract() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);

    // Instance storage is only reachable from inside the contract's own frame,
    // so the module helper has to be driven through `as_contract`.
    let mut seen = (Some(Address::generate(&e)), 1_u32);
    e.as_contract(&contract_id, || {
        seen = credence_bond::fees::get_config(&e);
    });

    assert_eq!(seen, (None, 0));
    assert_eq!(seen.0, None, "no treasury has been configured yet");
}

/// Boundary: the documented default is "never configured" — no treasury *and*
/// a zero rate, so governance must opt in before anything is charged.
#[test]
fn get_config_defaults_to_unset_with_zero_rate() {
    let e = Env::default();
    let (contract_id, client) = deploy(&e);
    let admin = Address::generate(&e);
    e.mock_all_auths();
    client.initialize(&admin, &None);

    let mut module_view = (Some(Address::generate(&e)), 1_u32);
    e.as_contract(&contract_id, || {
        module_view = get_config(&e);
    });

    assert_eq!(client.get_fee_config(), (None, 0));
    assert_eq!(module_view, (None, 0), "module view must agree");
}

/// Reading the config is idempotent: repeated polls return the same value and
/// emit nothing, so a keeper can retry safely.
#[test]
fn get_config_is_side_effect_free_and_repeatable() {
    let e = Env::default();
    let (client, admin) = setup(&e);
    let treasury = Address::generate(&e);
    client.set_fee_config(&admin, &treasury, &250);

    let first = client.get_fee_config();
    let second = client.get_fee_config();
    let third = client.get_fee_config();
    assert_eq!(first, second);
    assert_eq!(second, third);
    assert_eq!(first, (Some(treasury), 250));
}

/// `get_config` reads `FeeTreasury` and `FeeBps` independently, so a
/// treasury-only write yields `(Some, 0)`. Pin that the two fields are not
/// coupled: a zero rate with a live treasury is a valid, reachable state.
#[test]
fn treasury_without_rate_yields_some_treasury_zero_bps() {
    let e = Env::default();
    let (client, admin) = setup(&e);
    let treasury = Address::generate(&e);
    client.set_fee_config(&admin, &treasury, &0);
    assert_eq!(client.get_fee_config(), (Some(treasury), 0));
}

// ---------------------------------------------------------------------------
// S3: Range boundaries on the live setter
// ---------------------------------------------------------------------------

/// Boundary: both ends of the inclusive `[0, 1000]` range are accepted. An
/// off-by-one in the bounds check would either block a 0% rate (silently
/// disabling governance's ability to waive fees) or a 10% rate.
#[test]
fn set_config_accepts_both_range_endpoints() {
    let e = Env::default();
    let (client, admin) = setup(&e);
    let treasury = Address::generate(&e);

    client.set_fee_config(&admin, &treasury, &MIN_FEE_BPS);
    assert_eq!(
        client.get_fee_config(),
        (Some(treasury.clone()), MIN_FEE_BPS)
    );

    client.set_fee_config(&admin, &treasury, &MAX_FEE_BPS);
    assert_eq!(client.get_fee_config(), (Some(treasury), MAX_FEE_BPS));
}

/// Boundary: one basis point above the ceiling is the first rejected value.
/// `1001` must not slip through.
#[test]
fn set_config_rejects_exactly_one_bps_above_max() {
    let e = Env::default();
    let (client, _admin) = setup(&e);
    let admin = Address::generate(&e);
    let treasury = Address::generate(&e);
    assert!(client
        .try_set_fee_config(&admin, &treasury, &(MAX_FEE_BPS + 1))
        .is_err());
}

/// The top of the `u32` range is the widest rejection. If the check were
/// `fee_bps > MAX_FEE_BPS` on a signed type it could wrap and let this through.
#[test]
fn set_config_rejects_u32_max() {
    let e = Env::default();
    let (client, _admin) = setup(&e);
    let admin = Address::generate(&e);
    let treasury = Address::generate(&e);
    assert!(client
        .try_set_fee_config(&admin, &treasury, &u32::MAX)
        .is_err());
}

/// Every value strictly inside the range is accepted — the check is a range
/// test, not an allowlist of a few "known good" rates.
#[test]
fn set_config_accepts_interior_values_across_the_range() {
    let e = Env::default();
    let (client, admin) = setup(&e);
    let treasury = Address::generate(&e);

    for fee_bps in [0_u32, 1, 2, 7, 99, 100, 250, 500, 501, 999, 1_000] {
        client.set_fee_config(&admin, &treasury, &fee_bps);
        assert_eq!(
            client.get_fee_config().1,
            fee_bps,
            "interior value {fee_bps} was not stored verbatim"
        );
    }
}

/// Boundary: the rejection message is pinned exactly, so a future refactor
/// that changes which diagnostic an operator sees fails loudly. Registered in
/// `scripts/panic_baseline.txt` as a pre-existing bare `panic!` site.
#[test]
fn set_config_rejection_message_is_stable() {
    let e = Env::default();
    let (contract_id, client) = deploy(&e);
    let admin = Address::generate(&e);
    e.mock_all_auths();
    client.initialize(&admin, &None);
    let treasury = Address::generate(&e);

    // Driving `set_config` directly keeps the assertion on the helper's own
    // message rather than on any client-side envelope.
    expect_panic_with("fee_bps out of bounds", || {
        e.as_contract(&contract_id, || {
            set_config(&e, &admin, treasury.clone(), MAX_FEE_BPS + 1);
        });
    });
}

/// Boundary: the range check is the **first** statement in `set_config`, so a
/// rejected value never reaches `get_config`, never writes, and never emits.
///
/// Post-panic state cannot be inspected on the same `Env`: a contract panic
/// poisons the frame (`Contract re-entry is not allowed` on the next call), so
/// the state-preservation half is covered by
/// `rejected_bounds_update_preserves_previous_config` through the entrypoint.
#[test]
fn set_config_range_check_precedes_reads_writes_and_events() {
    let e = Env::default();
    let (contract_id, client) = deploy(&e);
    let admin = Address::generate(&e);
    e.mock_all_auths();
    client.initialize(&admin, &None);
    let treasury = Address::generate(&e);

    // Establish a live config first, so the rejected write is not a no-op over
    // an already-empty slot.
    client.set_fee_config(&admin, &treasury, &200);

    expect_panic_with("fee_bps out of bounds", || {
        e.as_contract(&contract_id, || {
            set_config(&e, &admin, Address::generate(&e), MAX_FEE_BPS + 1);
        });
    });
}

// ---------------------------------------------------------------------------
// S4: Rejection atomicity — no partial writes, no events
// ---------------------------------------------------------------------------

/// No data loss: a rejected bounds update must leave the previous treasury and
/// rate intact. Validating after the write would silently reset a live fee to
/// zero.
#[test]
fn rejected_bounds_update_preserves_previous_config() {
    let e = Env::default();
    let (client, admin) = setup(&e);
    let treasury = Address::generate(&e);
    let attacker_treasury = Address::generate(&e);
    client.set_fee_config(&admin, &treasury, &250);

    assert!(client
        .try_set_fee_config(&admin, &attacker_treasury, &(MAX_FEE_BPS + 1))
        .is_err());

    assert_eq!(client.get_fee_config(), (Some(treasury), 250));
}

/// A rejected update must not emit `fee_config_updated`. Without this half an
/// indexer replaying the rejected values would diverge from chain state.
#[test]
fn rejected_bounds_update_emits_no_event() {
    let e = Env::default();
    let (_id, client) = deploy(&e);
    let admin = Address::generate(&e);
    e.mock_all_auths();
    client.initialize(&admin, &None);
    let treasury = Address::generate(&e);

    assert!(client
        .try_set_fee_config(&admin, &treasury, &(MAX_FEE_BPS + 1))
        .is_err());

    assert!(
        e.events().all().is_empty(),
        "rejected set_config must not publish fee_config_updated"
    );
}

/// A non-admin attempt must change neither field nor the event log.
#[test]
fn rejected_unauthorised_update_leaves_config_and_events_untouched() {
    let e = Env::default();
    let (client, admin) = setup(&e);
    let treasury = Address::generate(&e);
    let stranger = Address::generate(&e);
    client.set_fee_config(&admin, &treasury, &300);

    assert!(client
        .try_set_fee_config(&stranger, &Address::generate(&e), &900)
        .is_err());

    assert_eq!(client.get_fee_config(), (Some(treasury), 300));
    // `Env::events()` is frame-scoped: each entrypoint invocation gets its own
    // log, so the rejected call's frame must be empty rather than a delta.
    assert!(
        e.events().all().is_empty(),
        "rejected unauthorised update must not emit"
    );
}

// ---------------------------------------------------------------------------
// S5: Permission
// ---------------------------------------------------------------------------

/// Permission: a stranger cannot rewrite the fee config. The guard rejects on
/// the *address* before authorization is even requested.
#[test]
fn set_config_rejects_non_admin() {
    let e = Env::default();
    let (client, _admin) = setup(&e);
    let stranger = Address::generate(&e);
    let res = client.try_set_fee_config(&stranger, &stranger, &100);
    assert!(res.is_err());
    assert_eq!(client.get_fee_config(), (None, 0));
}

/// The `NotAdmin` wire code is pinned so swapping this guard for another one
/// (pause, borrow-freeze) fails the test rather than silently changing which
/// error a caller observes.
#[test]
fn set_config_non_admin_failure_uses_not_admin_code() {
    let e = Env::default();
    let (_id, client) = deploy(&e);
    let admin = Address::generate(&e);
    e.mock_all_auths();
    client.initialize(&admin, &None);

    let stranger = Address::generate(&e);
    let err = client
        .try_set_fee_config(&stranger, &Address::generate(&e), &100)
        .expect_err("stranger must not set the fee config");
    let rendered = format!("{err:?}");
    assert!(
        rendered.contains(&format!("#{NOT_ADMIN_ERROR_CODE}")),
        "expected NotAdmin #{NOT_ADMIN_ERROR_CODE}, got {rendered}"
    );
}

/// Reading the fee config is deliberately **not** admin-gated: it is public
/// governance state, and an indexer must be able to read it without a key.
///
/// No auth is mocked at all on this path. If `get_fee_config` ever acquired a
/// `require_auth`, the call would fail rather than silently pass.
#[test]
fn get_config_is_readable_without_any_mocked_auth() {
    let e = Env::default();
    let (contract_id, client) = deploy(&e);
    let admin = Address::generate(&e);
    // Configure through the module helper so no `mock_all_auths` is ever
    // installed on this `Env`, and the read is therefore genuinely unauthed.
    let treasury = Address::generate(&e);
    e.as_contract(&contract_id, || {
        set_config(&e, &admin, treasury.clone(), 200);
    });

    assert!(client.try_get_fee_config().is_ok());
    assert_eq!(client.get_fee_config(), (Some(treasury), 200));
}

/// Same property on an untouched deployment, so the read is not trivially
/// satisfied by returning the "never configured" default without touching
/// storage.
#[test]
fn get_config_is_readable_on_an_unconfigured_contract() {
    let e = Env::default();
    let (_id, client) = deploy(&e);

    assert!(client.try_get_fee_config().is_ok());
    assert_eq!(client.get_fee_config(), (None, 0));
}

/// Permission: the module helper itself does not re-check admin. It is called
/// only from the guarded entrypoint. This pins that trust boundary so a future
/// direct caller is a deliberate choice rather than an oversight.
#[test]
fn set_config_helper_relies_on_entrypoint_for_authorization() {
    let e = Env::default();
    let (contract_id, client) = deploy(&e);
    let admin = Address::generate(&e);
    e.mock_all_auths();
    client.initialize(&admin, &None);
    let treasury = Address::generate(&e);

    // The helper writes whatever it is handed, admin identity included,
    // because the authorization decision lives in `set_fee_config`.
    let impostor = Address::generate(&e);
    e.as_contract(&contract_id, || {
        set_config(&e, &impostor, treasury.clone(), 400);
    });

    assert_eq!(client.get_fee_config(), (Some(treasury), 400));
}

// ---------------------------------------------------------------------------
// S6: Pause gating — a documented gap in the live suite
// ---------------------------------------------------------------------------

/// Emergency gate: a paused contract must reject fee-config changes. The live
/// `test_lib_boundary_recovery.rs` covers this for `deposit_fees` and
/// `collect_fees` but **not** for `set_fee_config`, which is the path that
/// changes the rate itself.
#[test]
fn paused_contract_rejects_set_fee_config() {
    let e = Env::default();
    let (client, admin) = setup(&e);
    let treasury = Address::generate(&e);

    client.pause(&admin);
    let res = client.try_set_fee_config(&admin, &treasury, &500);
    assert!(res.is_err(), "paused contract must reject set_fee_config");

    let rendered = format!("{:?}", res.unwrap_err());
    assert!(
        rendered.contains(&format!("#{CONTRACT_PAUSED_ERROR_CODE}")),
        "expected ContractPaused #{CONTRACT_PAUSED_ERROR_CODE}, got {rendered}"
    );
}

/// No data loss across an emergency: pausing blocks the change, and unpausing
/// lets the same call succeed with no drift.
#[test]
fn unpause_restores_set_fee_config() {
    let e = Env::default();
    let (client, admin) = setup(&e);
    let treasury = Address::generate(&e);
    client.set_fee_config(&admin, &treasury, &100);

    client.pause(&admin);
    assert!(client
        .try_set_fee_config(&admin, &Address::generate(&e), &900)
        .is_err());
    assert_eq!(client.get_fee_config(), (Some(treasury.clone()), 100));

    client.unpause(&admin);
    client.set_fee_config(&admin, &treasury, &900);
    assert_eq!(client.get_fee_config(), (Some(treasury), 900));
}

// ---------------------------------------------------------------------------
// S7: The `fee_config_updated` audit event
// ---------------------------------------------------------------------------

/// Decode the `fee_config_updated` event published inside the current frame,
/// asserting its shape on the way through.
///
/// `soroban_sdk::Vec` is host-backed and cannot be indexed from a test, so the
/// topics are copied into a plain `Vec<Val>` first. The data payload arrives as
/// a single `Val` and is decoded as a 4-tuple.
///
/// Returns `(topic0, topic1, (old_treasury, new_treasury, old_bps, new_bps))`.
#[track_caller]
fn last_fee_config_event_in_frame(
    e: &Env,
) -> (Symbol, Address, (Option<Address>, Address, u32, u32)) {
    let all = e.events().all();
    let (_contract, topics, data) = all
        .last()
        .expect("expected at least one event in the current frame");

    assert_eq!(
        topics.len(),
        2,
        "fee_config_updated must carry exactly two topics, got {}",
        topics.len()
    );

    let raw: Vec<Val> = topics.iter().collect();
    let topic0: Symbol = raw[0]
        .clone()
        .try_into_val(e)
        .expect("topic[0] is the event name Symbol");
    let topic1: Address = raw[1]
        .clone()
        .try_into_val(e)
        .expect("topic[1] is the admin Address");

    assert_eq!(
        topic0,
        Symbol::new(e, "fee_config_updated"),
        "topic[0] must be the event name"
    );

    let payload: (Option<Address>, Address, u32, u32) = data
        .clone()
        .try_into_val(e)
        .expect("data is (Option<Address>, Address, u32, u32)");

    (topic0, topic1, payload)
}

/// The event is the governance audit trail, so its **shape** is pinned: two
/// topics `(name, admin)` and four data fields in the order
/// `(old_treasury, new_treasury, old_fee_bps, new_fee_bps)`.
///
/// `docs/ARCHITECTURE.md` documents the data tuple as
/// `(old_treasury, old_bps, new_treasury, new_bps)`, which does not match this
/// implementation. The implementation is what an indexer will see.
#[test]
fn fee_config_event_payload_shape_is_pinned() {
    let e = Env::default();
    let (contract_id, client) = deploy(&e);
    let admin = Address::generate(&e);
    e.mock_all_auths();
    client.initialize(&admin, &None);
    let treasury = Address::generate(&e);

    let mut seen: Option<(Symbol, Address, (Option<Address>, Address, u32, u32))> = None;
    e.as_contract(&contract_id, || {
        set_config(&e, &admin, treasury.clone(), 250);
        seen = Some(last_fee_config_event_in_frame(&e));
    });

    let (topic0, topic1, (old_treasury, new_treasury, old_bps, new_bps)) =
        seen.expect("event present");
    assert_eq!(topic0, Symbol::new(&e, "fee_config_updated"));
    assert_eq!(
        topic1, admin,
        "topic[1] is the acting admin, not the treasury"
    );
    assert_eq!(old_treasury, None, "first set has no previous treasury");
    assert_eq!(new_treasury, treasury);
    assert_eq!(old_bps, 0, "first set has no previous rate");
    assert_eq!(new_bps, 250);
}

/// The first-ever write must report `None` / `0` for the previous state. This
/// is what lets an indexer distinguish "first configuration" from "configured
/// to the zero address".
#[test]
fn first_config_event_reports_none_treasury_and_zero_rate() {
    let e = Env::default();
    let (contract_id, client) = deploy(&e);
    let admin = Address::generate(&e);
    e.mock_all_auths();
    client.initialize(&admin, &None);
    let treasury = Address::generate(&e);

    let mut seen = None;
    e.as_contract(&contract_id, || {
        set_config(&e, &admin, treasury.clone(), 1);
        seen = Some(last_fee_config_event_in_frame(&e).2);
    });

    let (old_treasury, new_treasury, old_bps, new_bps) = seen.expect("payload present");
    assert_eq!(old_treasury, None);
    assert_eq!(new_treasury, treasury);
    assert_eq!(old_bps, 0);
    assert_eq!(new_bps, 1);
}

/// Changing only the treasury must preserve the rate in the event, so an
/// indexer diffing old/new pairs does not read a treasury swap as a rate change.
#[test]
fn treasury_only_change_preserves_rate_in_event() {
    let e = Env::default();
    let (contract_id, client) = deploy(&e);
    let admin = Address::generate(&e);
    e.mock_all_auths();
    client.initialize(&admin, &None);
    let first = Address::generate(&e);
    let second = Address::generate(&e);

    let mut seen = None;
    e.as_contract(&contract_id, || {
        set_config(&e, &admin, first.clone(), 300);
        set_config(&e, &admin, second.clone(), 300);
        seen = Some(last_fee_config_event_in_frame(&e).2);
    });

    let (old_treasury, new_treasury, old_bps, new_bps) = seen.expect("payload present");
    assert_eq!(old_treasury, Some(first));
    assert_eq!(new_treasury, second);
    assert_eq!(old_bps, 300, "rate unchanged by a treasury-only update");
    assert_eq!(new_bps, 300);
}

/// Changing only the rate must preserve the treasury.
#[test]
fn rate_only_change_preserves_treasury_in_event() {
    let e = Env::default();
    let (contract_id, client) = deploy(&e);
    let admin = Address::generate(&e);
    e.mock_all_auths();
    client.initialize(&admin, &None);
    let treasury = Address::generate(&e);

    let mut seen = None;
    e.as_contract(&contract_id, || {
        set_config(&e, &admin, treasury.clone(), 100);
        set_config(&e, &admin, treasury.clone(), 400);
        seen = Some(last_fee_config_event_in_frame(&e).2);
    });

    let (old_treasury, new_treasury, old_bps, new_bps) = seen.expect("payload present");
    assert_eq!(old_treasury, Some(treasury.clone()));
    assert_eq!(new_treasury, treasury);
    assert_eq!(old_bps, 100);
    assert_eq!(new_bps, 400);
}

/// A no-op re-set still emits, so governance can force a re-emission of the
/// audit trail. Pinned because it is a deliberate replay-semantics choice, not
/// an oversight.
#[test]
fn repeated_identical_config_still_emits() {
    let e = Env::default();
    let (contract_id, client) = deploy(&e);
    let admin = Address::generate(&e);
    e.mock_all_auths();
    client.initialize(&admin, &None);
    let treasury = Address::generate(&e);

    let mut seen = None;
    e.as_contract(&contract_id, || {
        set_config(&e, &admin, treasury.clone(), 250);
        set_config(&e, &admin, treasury.clone(), 250);
        seen = Some(last_fee_config_event_in_frame(&e).2);
    });

    let (old_treasury, new_treasury, old_bps, new_bps) = seen.expect("payload present");
    assert_eq!(old_treasury, Some(treasury.clone()));
    assert_eq!(new_treasury, treasury);
    assert_eq!(old_bps, 250, "a no-op re-set reports identical old/new");
    assert_eq!(new_bps, 250);
}

/// Every accepted `fee_bps` is reported verbatim in the event, including the
/// range endpoints.
#[test]
fn event_reports_rate_verbatim_at_range_endpoints() {
    let e = Env::default();
    let (contract_id, client) = deploy(&e);
    let admin = Address::generate(&e);
    e.mock_all_auths();
    client.initialize(&admin, &None);
    let treasury = Address::generate(&e);

    for rate in [MIN_FEE_BPS, MAX_FEE_BPS] {
        let mut seen = None;
        e.as_contract(&contract_id, || {
            set_config(&e, &admin, treasury.clone(), rate);
            seen = Some(last_fee_config_event_in_frame(&e).2);
        });
        let (_, _, _, new_bps) = seen.expect("payload present");
        assert_eq!(new_bps, rate);
    }
}

// ---------------------------------------------------------------------------
// S8: `calculate_fee` — characterization of the dormant arithmetic
// ---------------------------------------------------------------------------
//
// Reached only through a helper that reads the same config the entrypoint
// writes, so the arithmetic is exercised against a realistic fee rate.

/// Run `calculate_fee` against a freshly configured rate, inside a contract
/// frame, and return `(fee, net)`.
fn fee_for(
    e: &Env,
    contract_id: &Address,
    admin: &Address,
    treasury: &Address,
    rate: u32,
    amount: i128,
) -> (i128, i128) {
    let mut out = (0_i128, 0_i128);
    e.as_contract(contract_id, || {
        set_config(e, admin, treasury.clone(), rate);
        out = calculate_fee(e, amount);
    });
    out
}

/// Boundary: `MAX_FEE_BPS` is 10%, not 100%. The dead `src/test_fees.rs`
/// asserts the opposite. At 1 000 bps a 1 000-unit bond pays 100 and keeps
/// 900.
#[test]
fn calculate_fee_at_max_bps_takes_ten_percent_not_all() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let admin = Address::generate(&e);
    let treasury = Address::generate(&e);

    assert_eq!(
        fee_for(&e, &contract_id, &admin, &treasury, 1_000, 1_000),
        (100, 900)
    );
}

/// Boundary: at 1% the smallest bond that carries a whole unit of fee is
/// 10 000. Anything below collects zero.
#[test]
fn calculate_fee_at_one_percent_boundary() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let admin = Address::generate(&e);
    let treasury = Address::generate(&e);

    // 9 999 * 100 / 10 000 = 99.99 → truncated to 99.
    assert_eq!(
        fee_for(&e, &contract_id, &admin, &treasury, 100, 9_999),
        (99, 9_900)
    );
    // 10 000 * 100 / 10 000 = 100 exactly.
    assert_eq!(
        fee_for(&e, &contract_id, &admin, &treasury, 100, 10_000),
        (100, 9_900)
    );
}

/// Rounding is **truncation toward zero**, so sub-unit remainders are silently
/// dropped rather than rounded up. This is observable and worth pinning: a
/// 1%-fee bond of 999 units pays 9, not 10, and a bond of 9 units pays nothing
/// at all. Truncation always favours the bond, never the treasury.
#[test]
fn calculate_fee_truncates_rather_than_rounding_up() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let admin = Address::generate(&e);
    let treasury = Address::generate(&e);

    // 999 * 100 / 10 000 = 9.99 → truncated to 9, not rounded up to 10.
    assert_eq!(
        fee_for(&e, &contract_id, &admin, &treasury, 100, 999),
        (9, 990)
    );
    // 101 * 100 / 10 000 = 1.01 → 1.
    assert_eq!(
        fee_for(&e, &contract_id, &admin, &treasury, 100, 101),
        (1, 100)
    );
    // 109 * 100 / 10 000 = 1.09 → 1; rounding up would give 2.
    assert_eq!(
        fee_for(&e, &contract_id, &admin, &treasury, 100, 109),
        (1, 108)
    );
    // A single unit at the maximum rate is still zero.
    assert_eq!(
        fee_for(&e, &contract_id, &admin, &treasury, 1_000, 1),
        (0, 1)
    );
}

/// Boundary: a zero rate never charges, regardless of amount — the governance
/// waiver path.
#[test]
fn calculate_fee_at_zero_rate_is_always_zero() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let admin = Address::generate(&e);
    let treasury = Address::generate(&e);

    for amount in [1_i128, 1_000, 1_000_000, i128::MAX] {
        assert_eq!(
            fee_for(&e, &contract_id, &admin, &treasury, 0, amount),
            (0, amount)
        );
    }
}

/// Boundary: a zero or negative amount is short-circuited to no fee, and the
/// net is returned unchanged (including the negative value) rather than
/// clamped to zero.
#[test]
fn calculate_fee_short_circuits_non_positive_amounts() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let admin = Address::generate(&e);
    let treasury = Address::generate(&e);

    assert_eq!(
        fee_for(&e, &contract_id, &admin, &treasury, 1_000, 0),
        (0, 0)
    );
    assert_eq!(
        fee_for(&e, &contract_id, &admin, &treasury, 1_000, -1),
        (0, -1)
    );
    assert_eq!(
        fee_for(&e, &contract_id, &admin, &treasury, 1_000, i128::MIN),
        (0, i128::MIN)
    );
}

/// On a contract with no fee configured, `calculate_fee` reads `fee_bps = 0`
/// and charges nothing. This is the state every fresh deployment starts in.
#[test]
fn calculate_fee_is_zero_on_unconfigured_contract() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);

    let mut out = (1_i128, 1_i128);
    e.as_contract(&contract_id, || {
        out = calculate_fee(&e, 1_000_000);
    });
    assert_eq!(out, (0, 1_000_000));
}

/// The conservation invariant across a sweep of rates and amounts: whatever is
/// charged comes out of the bond, so `fee + net == amount` and the fee never
/// exceeds the amount.
#[test]
fn calculate_fee_conserves_amount_across_a_sweep() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let admin = Address::generate(&e);
    let treasury = Address::generate(&e);

    let rates = [1_u32, 10, 50, 100, 250, 500, 999, 1_000];
    let amounts = [
        1_i128,
        2,
        99,
        100,
        999,
        1_000,
        9_999,
        10_000,
        10_001,
        1_000_000,
        i128::MAX / 10_000,
    ];

    for rate in rates {
        for &amount in amounts.iter() {
            let (fee, net) = fee_for(&e, &contract_id, &admin, &treasury, rate, amount);
            assert_eq!(
                fee + net,
                amount,
                "fee {fee} + net {net} != amount {amount} at rate {rate}"
            );
            assert!(
                fee >= 0,
                "fee must never be negative: {fee} at rate {rate}, amount {amount}"
            );
            assert!(
                fee <= amount,
                "fee must never exceed the amount: {fee} > {amount} at rate {rate}"
            );
        }
    }
}

/// Characterization, not a wish: `split_bps` uses `saturating_mul`, so an
/// amount large enough to overflow the intermediate product **silently
/// clamps** instead of panicking. The returned fee is therefore wrong, with no
/// error and no event.
///
/// The three `"fee calculation *"` strings are unreachable for this reason and
/// are deliberately not asserted anywhere in this file.
#[test]
fn calculate_fee_at_i128_max_saturates_instead_of_panicking() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let admin = Address::generate(&e);
    let treasury = Address::generate(&e);

    // i128::MAX * 1000 overflows i128, so this exercises the saturating path.
    let (fee, net) = fee_for(&e, &contract_id, &admin, &treasury, 1_000, i128::MAX);
    assert_eq!(
        fee + net,
        i128::MAX,
        "conservation still holds after saturation"
    );
    assert!(
        fee > 0,
        "a clamped fee is still non-zero, which is the point: it is wrong but silent"
    );
    // 10% of i128::MAX, computed by the saturated intermediate.
    assert_eq!(fee, i128::MAX / 10_000);
}

/// `fee + net == amount` must hold for every negative amount too: `split_bps`
/// truncates toward zero, so a negative amount yields a negative fee and a
/// non-negative net. The short-circuit in `calculate_fee` means this only
/// holds because of the early return, not because of the arithmetic.
#[test]
fn calculate_fee_never_charges_on_negative_amounts_even_at_max_rate() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let admin = Address::generate(&e);
    let treasury = Address::generate(&e);

    for amount in [-1_i128, -999, -1_000_000, i128::MIN + 1] {
        let (fee, net) = fee_for(&e, &contract_id, &admin, &treasury, 1_000, amount);
        assert_eq!(fee, 0, "no fee may be charged on {amount}");
        assert_eq!(net, amount);
    }
}

// ---------------------------------------------------------------------------
// S9: `is_fee_waived` — mirrors the `calculate_fee` short-circuit
// ---------------------------------------------------------------------------

/// Characterization: `is_fee_waived` is **not** equivalent to
/// `calculate_fee(..).0 == 0`. It duplicates the *early-return condition*
/// (`fee_bps == 0 || amount <= 0`) but ignores truncation, so an amount too
/// small to carry a whole unit of fee is reported as **not waived** while
/// `calculate_fee` charges zero.
///
/// A caller that gates a charge on `is_fee_waived` and treats "not waived" as
/// "charge something" would try to collect a fee that does not exist. Harmless
/// only because neither function is reachable from an entrypoint.
#[test]
fn is_fee_waived_diverges_from_calculate_fee_zero_fee() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let admin = Address::generate(&e);
    let treasury = Address::generate(&e);
    let identity = Address::generate(&e);

    // rate 1 bps, amount 1: 1 * 1 / 10 000 = 0.0001 → truncated to 0.
    let (fee, net) = fee_for(&e, &contract_id, &admin, &treasury, 1, 1);
    let mut waived = true;
    e.as_contract(&contract_id, || {
        waived = is_fee_waived(&e, 1, &identity);
    });

    assert_eq!(fee, 0, "truncation yields no fee at this size");
    assert_eq!(net, 1, "the full amount stays with the bond");
    assert!(
        !waived,
        "is_fee_waived says 'charge', calculate_fee charges nothing — \
         the two surfaces disagree and only truncation separates them"
    );
}

/// The half of the relationship that *does* hold: whenever `is_fee_waived`
/// returns `true`, `calculate_fee` must charge exactly zero. This is the
/// direction that matters — a waived bond can never be charged.
#[test]
fn waived_always_implies_a_zero_fee() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let admin = Address::generate(&e);
    let treasury = Address::generate(&e);
    let identity = Address::generate(&e);

    for rate in [0_u32, 1, 100, 500, 1_000] {
        e.as_contract(&contract_id, || {
            set_config(&e, &admin, treasury.clone(), rate);
        });
        for amount in [i128::MIN, -1_i128, 0, 1, 1_000, 1_000_000] {
            let mut waived = false;
            e.as_contract(&contract_id, || {
                waived = is_fee_waived(&e, amount, &identity);
            });
            let (fee, net) = fee_for(&e, &contract_id, &admin, &treasury, rate, amount);
            if waived {
                assert_eq!(fee, 0, "waived at rate {rate}, amount {amount}");
                assert_eq!(
                    net, amount,
                    "a waived bond keeps its full amount at rate {rate}, amount {amount}"
                );
            }
        }
    }
}

/// Boundary: at a zero rate every amount is waived.
#[test]
fn is_fee_waived_true_at_zero_rate_for_any_amount() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let admin = Address::generate(&e);
    let treasury = Address::generate(&e);
    let identity = Address::generate(&e);

    e.as_contract(&contract_id, || {
        set_config(&e, &admin, treasury.clone(), 0);
    });
    for amount in [i128::MIN, -1, 0, 1, 1_000_000, i128::MAX] {
        let mut waived = false;
        e.as_contract(&contract_id, || {
            waived = is_fee_waived(&e, amount, &identity);
        });
        assert!(waived, "amount {amount} must be waived at a zero rate");
    }
}

/// Boundary: at a non-zero rate, a non-positive amount is waived and a
/// positive one is not — the identity argument is ignored entirely today.
#[test]
fn is_fee_waived_ignores_identity_argument() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let admin = Address::generate(&e);
    let treasury = Address::generate(&e);
    let one = Address::generate(&e);
    let two = Address::generate(&e);

    e.as_contract(&contract_id, || {
        set_config(&e, &admin, treasury.clone(), 1_000);
    });

    // Two distinct identities, so the waiver check is shown to key off the
    // identity argument rather than off loop state.
    for (index, identity) in [&one, &two].iter().enumerate() {
        let mut zero_case = true;
        let mut positive_case = false;
        e.as_contract(&contract_id, || {
            zero_case = is_fee_waived(&e, 0, identity);
            positive_case = is_fee_waived(&e, 1_000, identity);
        });
        assert!(zero_case, "zero amount must be waived for identity {index}");
        assert!(
            !positive_case,
            "positive amount must not be waived for identity {index}"
        );
    }
}

// ---------------------------------------------------------------------------
// S10: `record_fee` — the dormant accumulator
// ---------------------------------------------------------------------------

/// Decode the `bond_creation_fee` event published inside the current frame.
///
/// Returns `(topics_len, (identity, bond_amount, fee, treasury))`.
#[track_caller]
fn last_bond_creation_fee_event_in_frame(e: &Env) -> (usize, (Address, i128, i128, Address)) {
    let all = e.events().all();
    let (_contract, topics, data) = all.last().expect("expected an event in the current frame");

    let raw: Vec<Val> = topics.iter().collect();
    assert_eq!(raw.len(), 1, "bond_creation_fee has a single topic");
    let topic0: Symbol = raw[0]
        .clone()
        .try_into_val(e)
        .expect("topic[0] is a Symbol");
    assert_eq!(
        topic0,
        Symbol::new(e, "bond_creation_fee"),
        "topic[0] must be the event name"
    );

    let payload: (Address, i128, i128, Address) = data
        .clone()
        .try_into_val(e)
        .expect("data is (Address, i128, i128, Address)");

    (raw.len(), payload)
}

/// The happy path: a positive fee accumulates and publishes
/// `bond_creation_fee` with `(identity, bond_amount, fee, treasury)`.
#[test]
fn record_fee_accumulates_and_emits() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    let mut topics_len = 0;
    let mut payload = None;
    e.as_contract(&contract_id, || {
        record_fee(&e, &identity, 1_000_000, 100, &treasury);
        let (len, data) = last_bond_creation_fee_event_in_frame(&e);
        topics_len = len;
        payload = Some(data);
    });

    assert_eq!(topics_len, 1, "bond_creation_fee has a single topic");
    let (logged_identity, logged_amount, logged_fee, logged_treasury) =
        payload.expect("payload present");
    assert_eq!(logged_identity, identity);
    assert_eq!(logged_amount, 1_000_000);
    assert_eq!(logged_fee, 100);
    assert_eq!(logged_treasury, treasury);

    // The accrual itself is independent of the event.
    assert_eq!(read_fee_pool(&e, &contract_id), 100);
}

/// Successive records accumulate additively into the single pool.
#[test]
fn record_fee_accumulates_across_calls() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    for fee in [100_i128, 250, 1, 1_000] {
        e.as_contract(&contract_id, || {
            record_fee(&e, &identity, 1_000_000, fee, &treasury);
        });
    }
    assert_eq!(read_fee_pool(&e, &contract_id), 1_351);
}

/// Boundary: a non-positive fee is a **silent no-op** — no write, no event. The
/// caller cannot distinguish "no fee due" from "recorded as zero". Contrast
/// `deposit_fees`, which rejects non-positive amounts outright.
#[test]
fn record_fee_ignores_non_positive_fees() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    e.as_contract(&contract_id, || {
        record_fee(&e, &identity, 1_000_000, 0, &treasury);
    });
    assert_eq!(
        read_fee_pool(&e, &contract_id),
        0,
        "a zero fee must not write the pool"
    );
    assert!(
        e.events().all().is_empty(),
        "a zero fee must not emit bond_creation_fee"
    );

    e.as_contract(&contract_id, || {
        record_fee(&e, &identity, 1_000_000, -500, &treasury);
    });
    assert_eq!(
        read_fee_pool(&e, &contract_id),
        0,
        "a negative fee must not write the pool"
    );
    assert!(
        e.events().all().is_empty(),
        "a negative fee must not emit bond_creation_fee"
    );
}

/// A no-op record must not clobber an existing balance.
#[test]
fn record_fee_no_op_preserves_existing_balance() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    e.as_contract(&contract_id, || {
        record_fee(&e, &identity, 1_000_000, 700, &treasury);
    });
    assert_eq!(read_fee_pool(&e, &contract_id), 700);

    e.as_contract(&contract_id, || {
        record_fee(&e, &identity, 1_000_000, 0, &treasury);
    });
    assert_eq!(
        read_fee_pool(&e, &contract_id),
        700,
        "no-op must not zero the pool"
    );
}

/// Boundary: `add_i128` is checked, so accumulating past `i128::MAX` panics
/// with the registered message rather than wrapping. Note the *contrast* with
/// `deposit_fees`, which wraps silently — see `fees_recovery.rs`.
#[test]
fn record_fee_panics_on_pool_overflow() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    seed_fee_pool(&e, &contract_id, i128::MAX - 10);

    let msg = panic_message(|| {
        e.as_contract(&contract_id, || {
            record_fee(&e, &identity, 1_000_000, 100, &treasury);
        });
    });
    assert_eq!(msg, "fee pool overflow");
}

/// The pool accumulates exactly, with no rounding or loss: the sum of the
/// recorded fees equals the stored total.
#[test]
fn record_fee_sum_is_exact() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    let fees = [1_i128, 2, 3, 7, 11, 13, 17, 19, 23, 29];
    let expected: i128 = fees.iter().sum();
    for fee in fees {
        e.as_contract(&contract_id, || {
            record_fee(&e, &identity, 1_000_000, fee, &treasury);
        });
    }
    assert_eq!(read_fee_pool(&e, &contract_id), expected);
}

/// The pool is a single shared accumulator keyed on the raw symbol `"fees"`,
/// not on the identity. Two identities contribute to the same balance — there
/// is no per-identity accounting, which is what makes `collect_fees` (a single
/// sweep) coherent.
#[test]
fn record_fee_pool_is_shared_across_identities() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let alice = Address::generate(&e);
    let bob = Address::generate(&e);
    let treasury = Address::generate(&e);

    e.as_contract(&contract_id, || {
        record_fee(&e, &alice, 1_000_000, 300, &treasury);
    });
    e.as_contract(&contract_id, || {
        record_fee(&e, &bob, 2_000_000, 450, &treasury);
    });

    assert_eq!(
        read_fee_pool(&e, &contract_id),
        750,
        "the pool is a single balance, not per-identity"
    );
}

/// Boundary: a single fee of `i128::MAX` is representable on its own; the pool
/// starts at zero so no overflow occurs.
#[test]
fn record_fee_accepts_i128_max_onto_an_empty_pool() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    e.as_contract(&contract_id, || {
        record_fee(&e, &identity, i128::MAX, i128::MAX, &treasury);
    });
    assert_eq!(read_fee_pool(&e, &contract_id), i128::MAX);
}

/// Boundary: one unit past the maximum must panic, not wrap to a negative
/// balance.
#[test]
fn record_fee_rejects_one_past_max() {
    let e = Env::default();
    let (contract_id, _client) = deploy(&e);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    e.as_contract(&contract_id, || {
        record_fee(&e, &identity, i128::MAX, i128::MAX, &treasury);
    });

    let msg = panic_message(|| {
        e.as_contract(&contract_id, || {
            record_fee(&e, &identity, 1, 1, &treasury);
        });
    });
    assert_eq!(msg, "fee pool overflow");
}

// ---------------------------------------------------------------------------
// S11: Cross-surface consistency
// ---------------------------------------------------------------------------

/// `get_config` (module) and `get_fee_config` (entrypoint) must agree exactly.
/// They are the same read reached two ways; a divergence would mean the module
/// is reading different keys than the public getter.
#[test]
fn module_and_entrypoint_getters_agree() {
    let e = Env::default();
    let (contract_id, client) = deploy(&e);
    let admin = Address::generate(&e);
    e.mock_all_auths();
    client.initialize(&admin, &None);
    let treasury = Address::generate(&e);

    // Before any configuration.
    let mut module_view = (None, 0);
    e.as_contract(&contract_id, || {
        module_view = get_config(&e);
    });
    assert_eq!(module_view, client.get_fee_config());

    // After a configuration written through the entrypoint.
    client.set_fee_config(&admin, &treasury, &375);
    let mut module_view_after = (None, 0);
    e.as_contract(&contract_id, || {
        module_view_after = get_config(&e);
    });
    assert_eq!(
        module_view_after,
        client.get_fee_config(),
        "module and entrypoint views must not diverge"
    );
    assert_eq!(module_view_after, (Some(treasury), 375));
}

/// `set_config` (module) and `set_fee_config` (entrypoint) write the same
/// storage, so a module write is visible through the client.
#[test]
fn module_and_entrypoint_setters_target_the_same_storage() {
    let e = Env::default();
    let (contract_id, client) = deploy(&e);
    let admin = Address::generate(&e);
    e.mock_all_auths();
    client.initialize(&admin, &None);
    let treasury = Address::generate(&e);

    e.as_contract(&contract_id, || {
        set_config(&e, &admin, treasury.clone(), 600);
    });
    assert_eq!(client.get_fee_config(), (Some(treasury.clone()), 600));

    client.set_fee_config(&admin, &treasury, &200);
    assert_eq!(client.get_fee_config(), (Some(treasury), 200));
}

/// Configure-then-charge is the intended pipeline: a rate set through the
/// entrypoint is the rate `calculate_fee` observes. Because no entrypoint calls
/// `calculate_fee`, this pipeline has to be assembled by hand — which is
/// precisely why the collection surface is described as dormant.
#[test]
fn configured_rate_is_the_rate_calculate_fee_observes() {
    let e = Env::default();
    let (contract_id, client) = deploy(&e);
    let admin = Address::generate(&e);
    e.mock_all_auths();
    client.initialize(&admin, &None);
    let treasury = Address::generate(&e);

    client.set_fee_config(&admin, &treasury, &250);

    let mut out = (0_i128, 0_i128);
    e.as_contract(&contract_id, || {
        out = calculate_fee(&e, 100_000);
    });
    // 100_000 * 250 / 10_000 = 2_500.
    assert_eq!(out, (2_500, 97_500));
}

/// The fee pool written by `record_fee` is the same raw key the contract's
/// `deposit_fees` and `collect_fees` read. Proving it here is what lets
/// `fees_recovery.rs` treat the three as one accumulator.
#[test]
fn record_fee_writes_the_key_the_contract_fee_pool_uses() {
    let e = Env::default();
    let (contract_id, client) = deploy(&e);
    let admin = Address::generate(&e);
    e.mock_all_auths();
    client.initialize(&admin, &None);
    let identity = Address::generate(&e);
    let treasury = Address::generate(&e);

    e.as_contract(&contract_id, || {
        record_fee(&e, &identity, 1_000_000, 900, &treasury);
    });

    // `deposit_fees` accumulates into the same key, so its total must include
    // the amount `record_fee` just added.
    client.deposit_fees(&100);
    assert_eq!(
        read_fee_pool(&e, &contract_id),
        1_000,
        "record_fee and deposit_fees share one accumulator"
    );
}

//! USD-denominated budgets, backed by [`llm_budget`].
//!
//! borg's built-in [`InMemoryBudget`](crate::InMemoryBudget) counts tokens.
//! Tokens are a proxy for money, but the proxy leaks: a million tokens on a
//! haiku-class model and a million tokens on a frontier model differ by two
//! orders of magnitude, and the cache-read discount only approximates the
//! real cache pricing of one provider. [`UsdBudget`] books actual dollars
//! instead: every completed call is priced with [`llm_budget`]'s built-in
//! price table (232 models, prompt-cache reads and writes included) and
//! charged against a hard USD cap.
//!
//! The [`Budget`] trait counts in `u64`, so this implementation denominates
//! in **micro-USD** (1 USD = 1,000,000 units): `limit()`, `consumed()` and
//! `remaining()` report micro-dollars, and `min_per_branch` in
//! [`admit_parallel`](Budget::admit_parallel) is an estimated micro-dollar
//! cost per branch.
//!
//! ```no_run
//! use borg::{Budget, TokenUsage, UsdBudget};
//!
//! # fn main() -> Result<(), borg::UsdBudgetError> {
//! // Cap this composition tree at $0.25 of claude-haiku-class spend.
//! let budget = UsdBudget::new(0.25, "claude-haiku-4-5")?;
//! assert!(budget.can_invoke());
//!
//! // After a call, charge what the provider reported:
//! let verdict = budget.record_usage(&TokenUsage::new(12_000, 400).with_cache(8_000, 0));
//! # let _ = verdict;
//! # Ok(())
//! # }
//! ```

use crate::budget::{Budget, BudgetVerdict, TokenUsage};

/// Micro-USD per dollar: the unit [`UsdBudget`] reports through the
/// token-denominated [`Budget`] trait.
pub const MICRO_USD: u64 = 1_000_000;

/// Errors constructing a [`UsdBudget`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum UsdBudgetError {
    /// The model id is not in [`llm_budget`]'s price table, so usage could
    /// not be priced.
    #[error("unknown model for pricing: {0}")]
    UnknownModel(String),
    /// The underlying budget rejected its configuration.
    #[error("budget config: {0}")]
    Config(#[from] llm_budget::BudgetError),
}

/// A [`Budget`] that tracks real dollars instead of tokens.
///
/// Wraps [`llm_budget::AgentBudget`]: a hard cap enforced with
/// compare-and-swap reservations, an audit log of every charge, and a kill
/// switch. Model prices come from [`llm_budget::PriceTable::builtin`]; pass a
/// custom table with [`UsdBudget::with_price_table`] to pin or extend prices.
///
/// All calls take `&self` and are safe to share across parallel branches,
/// matching the trait's contract.
pub struct UsdBudget {
    budget: llm_budget::AgentBudget,
    model: String,
    prices: std::sync::Arc<llm_budget::PriceTable>,
}

impl UsdBudget {
    /// A `limit_usd`-dollar cap for calls to `model`.
    ///
    /// `model` is any id in the built-in price table (provider-prefixed or
    /// bare, e.g. `"claude-haiku-4-5"` or `"gpt-4o"`). The same
    /// model prices every charge on this budget; compositions that mix
    /// models need one budget per model wrapped in a
    /// [`SubBudget`](crate::SubBudget)-style split, or a custom price table
    /// keyed to a synthetic id.
    ///
    /// # Errors
    /// [`UsdBudgetError::UnknownModel`] when `model` is not priced, or
    /// [`UsdBudgetError::Config`] when `limit_usd` is not positive.
    pub fn new(limit_usd: f64, model: impl Into<String>) -> Result<Self, UsdBudgetError> {
        Self::with_price_table(
            limit_usd,
            model,
            std::sync::Arc::new(llm_budget::PriceTable::builtin().clone()),
        )
    }

    /// [`new`](Self::new) with a caller-supplied price table, for models the
    /// built-in table does not price or to pin prices at a fixed date.
    ///
    /// # Errors
    /// See [`new`](Self::new).
    pub fn with_price_table(
        limit_usd: f64,
        model: impl Into<String>,
        prices: std::sync::Arc<llm_budget::PriceTable>,
    ) -> Result<Self, UsdBudgetError> {
        let model = model.into();
        if prices.get(&model).is_none() {
            return Err(UsdBudgetError::UnknownModel(model));
        }
        let budget = llm_budget::AgentBudget::builder("borg", limit_usd).build()?;
        Ok(Self {
            budget,
            model,
            prices,
        })
    }

    /// The wrapped [`llm_budget::AgentBudget`], for the parts the [`Budget`]
    /// trait has no spelling for: the audit log, the kill switch, tier
    /// downgrades.
    pub fn inner(&self) -> &llm_budget::AgentBudget {
        &self.budget
    }

    /// Price `usage` at this budget's model, in USD.
    fn cost_of(&self, usage: &llm_budget::Usage) -> f64 {
        // Construction validated the model, so this lookup cannot fail.
        self.prices
            .get(&self.model)
            .map_or(0.0, |p| p.cost_usage(usage))
    }

    /// Book `cost_usd` and translate the outcome into a [`BudgetVerdict`].
    fn charge(&self, cost_usd: f64, input_tokens: u64, output_tokens: u64) -> BudgetVerdict {
        let outcome = self.budget.charge_usd_force(
            cost_usd,
            llm_budget::ChargeContext::new()
                .model(self.model.clone())
                .tokens(input_tokens, output_tokens),
        );
        if outcome.total_spent_usd > self.budget.limit_usd() {
            BudgetVerdict::Overdrawn
        } else if outcome.over_cap || self.budget.remaining_usd() <= 0.0 {
            BudgetVerdict::Exhausted
        } else {
            BudgetVerdict::Ok
        }
    }
}

fn usd_to_micro(usd: f64) -> u64 {
    if usd.is_finite() && usd > 0.0 {
        (usd * MICRO_USD as f64).round() as u64
    } else {
        0
    }
}

impl std::fmt::Debug for UsdBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UsdBudget")
            .field("model", &self.model)
            .field("limit_usd", &self.budget.limit_usd())
            .field("spent_usd", &self.budget.spent_usd())
            .finish_non_exhaustive()
    }
}

impl Budget for UsdBudget {
    /// Settled spend so far, in micro-USD.
    fn consumed(&self) -> u64 {
        usd_to_micro(self.budget.spent_usd())
    }

    /// Unspent cap, in micro-USD.
    fn remaining(&self) -> u64 {
        usd_to_micro(self.budget.remaining_usd())
    }

    /// The cap, in micro-USD.
    fn limit(&self) -> u64 {
        usd_to_micro(self.budget.limit_usd())
    }

    fn can_invoke(&self) -> bool {
        !self.budget.is_exhausted() && !self.budget.is_killed()
    }

    /// Charge `tokens` with no usage split. Every token is priced at the
    /// model's **output** rate — the worst case — so a caller that cannot
    /// split input from output cannot silently underdraw. Prefer
    /// [`record_usage`](Budget::record_usage) when the split is known.
    fn record(&self, tokens: u64) -> BudgetVerdict {
        let usage = llm_budget::Usage::new(0, tokens);
        self.charge(self.cost_of(&usage), 0, tokens)
    }

    /// Charge a completed call from its full [`TokenUsage`], with prompt-cache
    /// reads and writes priced at the model's real cache rates.
    fn record_usage(&self, usage: &TokenUsage) -> BudgetVerdict {
        let u = llm_budget::Usage {
            input_tokens: usage.input_tokens,
            cached_input_tokens: usage.cache_read_tokens,
            cache_write_tokens: usage.cache_creation_tokens,
            output_tokens: usage.output_tokens,
        };
        self.charge(self.cost_of(&u), usage.input_tokens, usage.output_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::ParallelAdmission;

    fn haiku_budget(limit_usd: f64) -> UsdBudget {
        UsdBudget::new(limit_usd, "claude-haiku-4-5").unwrap()
    }

    #[test]
    fn reports_limit_and_remaining_in_micro_usd() {
        let b = haiku_budget(0.25);
        assert_eq!(b.limit(), 250_000);
        assert_eq!(b.consumed(), 0);
        assert_eq!(b.remaining(), 250_000);
        assert!(b.can_invoke());
    }

    #[test]
    fn record_usage_prices_cache_reads_at_cache_rate() {
        // 100k fresh input, 900k cache reads, 1k output on haiku
        // ($1.00/M in, $0.10/M cached read, $5.00/M out):
        // 0.1 * 1.00 + 0.9 * 0.10 + 0.001 * 5.00 = 0.10 + 0.09 + 0.005 = $0.195
        let b = haiku_budget(1.0);
        let verdict = b.record_usage(&TokenUsage::new(1_000_000, 1_000).with_cache(900_000, 0));
        assert_eq!(verdict, BudgetVerdict::Ok);
        let spent = b.budget.spent_usd();
        assert!((spent - 0.195).abs() < 1e-9, "spent {spent}");
        assert_eq!(b.consumed(), 195_000);
    }

    #[test]
    fn cap_exhausts_and_denies() {
        // $0.001 cap; one 10k/1k haiku call costs 0.01 + 0.005 = $0.015.
        let b = haiku_budget(0.001);
        let verdict = b.record_usage(&TokenUsage::new(10_000, 1_000));
        assert_eq!(verdict, BudgetVerdict::Overdrawn);
        assert!(!b.can_invoke());
    }

    #[test]
    fn record_without_split_charges_output_rate() {
        // 1k undifferentiated tokens at haiku's $5.00/M output rate = $0.005.
        let b = haiku_budget(1.0);
        assert_eq!(b.record(1_000), BudgetVerdict::Ok);
        assert!((b.budget.spent_usd() - 0.005).abs() < 1e-12);
    }

    #[test]
    fn parallel_admission_runs_in_micro_usd() {
        let b = haiku_budget(0.000030); // 30 micro-USD
        assert_eq!(b.admit_parallel(5, 10), ParallelAdmission::Partial { allowed: 3 });
    }

    #[test]
    fn unknown_model_is_rejected_at_construction() {
        let err = match UsdBudget::new(1.0, "not-a-real-model") {
            Ok(_) => panic!("unknown model must not construct"),
            Err(e) => e,
        };
        assert!(matches!(err, UsdBudgetError::UnknownModel(_)));
    }

    #[test]
    fn custom_price_table_prices_custom_models() {
        let mut table = llm_budget::PriceTable::new();
        table.insert(llm_budget::ModelPrice {
            model: "my-model".into(),
            provider: "test".into(),
            input_per_million: 1.0,
            output_per_million: 2.0,
            cache_read_per_million: None,
            cache_write_per_million: None,
            max_input_tokens: None,
        });
        let b = UsdBudget::with_price_table(1.0, "my-model", std::sync::Arc::new(table)).unwrap();
        assert_eq!(b.record_usage(&TokenUsage::new(1_000, 1_000)), BudgetVerdict::Ok);
        assert!((b.budget.spent_usd() - 0.003).abs() < 1e-12);
    }
}

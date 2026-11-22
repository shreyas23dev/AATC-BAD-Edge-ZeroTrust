//! Policy decision type.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Authorization decision produced by the AATC-BAD engine per telemetry frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyDecision {
    /// Trust >= T_allow AND cumulative debt < K_allow
    Allow,
    /// Trust in [T_quar, T_allow)
    Verify,
    /// Trust < T_quar OR debt >= K_allow
    Block,
}

impl fmt::Display for PolicyDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolicyDecision::Allow  => write!(f, "ALLOW"),
            PolicyDecision::Verify => write!(f, "VERIFY"),
            PolicyDecision::Block  => write!(f, "BLOCK"),
        }
    }
}

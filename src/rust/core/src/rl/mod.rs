// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
//! RL Optimizer Module
//!
//! Reinforcement Learning for Autonomous Mix Optimization (Blueprint Section 6.4)
//! Implements multiple reward functions and PPO-style policy optimization.
//!
//! Modular Design:
//! - traits.rs: IRewardProvider, IDataProvider interfaces
//! - concrete_provider.rs: Concrete cartridge implementation
//! - ppo.rs: Material-agnostic PPO agent

pub mod concrete_provider;
pub mod guardrails;
mod ppo;
mod reward;
pub mod state;
pub mod traits;

pub use concrete_provider::ConcreteRewardProvider;
pub use guardrails::{GuardrailEngine, GuardrailValidation, PhysicsGuardrails, ViolationSeverity};
pub use ppo::{PPOAgent, PPOConfig};
pub use reward::{RewardComponents, RewardConfig, RewardFunction, RewardType};
pub use state::{RLAction, RLState};
pub use traits::{CartridgeInfo, IDataProvider, IRewardProvider, IScienceCartridge, MaterialData};

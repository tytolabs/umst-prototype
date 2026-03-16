// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, Studio Tyto
// SPDX-License-Identifier: MIT

//! # Kleisli Arrow — Admissibility Monad for UMST State Transitions
//!
//! Formalises agent actions as morphisms in the Kleisli category over the
//! admissibility monad M. Sequential composition of Kleisli arrows preserves
//! thermodynamic admissibility by construction:
//!
//!   (f ● g)(x) = μ(M(f)(g(x)))  ⟹  D_int((f ● g)(x)) ≥ 0
//!
//! This is the Rust implementation backing Paper 3's categorical safety
//! guarantee. The formal proofs (monad laws, subject reduction, N-step
//! admissibility) live in `umst-formal/` (Agda, Coq, Lean 4).

use crate::tensors::MixTensor;

/// Result of a thermodynamic gate check on a state transition.
#[derive(Debug, Clone)]
pub struct AdmissibilityResult {
    pub admissible: bool,
    pub dissipation: f32,
    pub violation: Option<String>,
}

/// The admissibility monad wraps a value with its gate status.
/// M(A) = (A, AdmissibilityResult)
#[derive(Debug, Clone)]
pub struct Admissible<A: Clone> {
    pub value: A,
    pub result: AdmissibilityResult,
}

impl<A: Clone> Admissible<A> {
    /// Monadic unit (η): lift a value into the admissibility monad.
    /// The trivial self-transition is always admissible.
    pub fn pure(value: A) -> Self {
        Admissible {
            value,
            result: AdmissibilityResult {
                admissible: true,
                dissipation: 0.0,
                violation: None,
            },
        }
    }

    /// Monadic bind (>>=): compose with a Kleisli arrow.
    /// Short-circuits on inadmissible intermediate states.
    pub fn bind<B: Clone, F>(self, f: F) -> Admissible<B>
    where
        F: FnOnce(A) -> Admissible<B>,
    {
        if !self.result.admissible {
            return Admissible {
                value: f(self.value).value,
                result: self.result,
            };
        }
        f(self.value)
    }

    /// Monadic join (μ): flatten M(M(A)) → M(A).
    pub fn join(nested: Admissible<Admissible<A>>) -> Admissible<A> {
        if !nested.result.admissible {
            Admissible {
                value: nested.value.value,
                result: nested.result,
            }
        } else {
            nested.value
        }
    }
}

/// A Kleisli arrow: a function A → M(B) in the admissibility monad.
/// Agent actions, physics predictions, and gate checks are all Kleisli arrows.
pub struct KleisliArrow<A: Clone, B: Clone> {
    pub name: String,
    arrow: Box<dyn Fn(A) -> Admissible<B> + Send + Sync>,
}

impl<A: Clone, B: Clone> KleisliArrow<A, B> {
    pub fn new<F>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(A) -> Admissible<B> + Send + Sync + 'static,
    {
        KleisliArrow {
            name: name.into(),
            arrow: Box::new(f),
        }
    }

    pub fn run(&self, input: A) -> Admissible<B> {
        (self.arrow)(input)
    }
}

/// Compose two Kleisli arrows sequentially: (f ● g)(x) = f(x) >>= g
/// The resulting arrow is A → M(C) and preserves admissibility.
pub fn kleisli_compose_pair<A, B, C>(
    f: impl Fn(A) -> Admissible<B> + Send + Sync + 'static,
    g: impl Fn(B) -> Admissible<C> + Send + Sync + 'static,
    name: impl Into<String>,
) -> KleisliArrow<A, C>
where
    A: Clone + 'static,
    B: Clone + 'static,
    C: Clone + 'static,
{
    KleisliArrow::new(name, move |a: A| {
        let mb = f(a);
        mb.bind(|b| g(b))
    })
}

/// Convenient Kleisli pipeline builder for sequential agent actions.
pub struct KleisliPipeline {
    pub name: String,
    pub steps: Vec<String>,
}

impl KleisliPipeline {
    pub fn new(name: impl Into<String>) -> Self {
        KleisliPipeline {
            name: name.into(),
            steps: Vec::new(),
        }
    }

    /// Run a sequence of MixTensor → MixTensor Kleisli arrows.
    pub fn run_sequence(
        &self,
        initial: MixTensor,
        arrows: &[&KleisliArrow<MixTensor, MixTensor>],
    ) -> Admissible<MixTensor> {
        let mut current = Admissible::pure(initial);

        for arrow in arrows.iter() {
            if !current.result.admissible {
                break;
            }
            current = current.bind(|state| arrow.run(state));
        }

        current
    }
}

/// Create a gate-checking Kleisli arrow from a predicate on MixTensor.
pub fn gate_arrow(
    name: impl Into<String>,
    check: impl Fn(&MixTensor) -> (bool, f32, Option<String>) + Send + Sync + 'static,
) -> KleisliArrow<MixTensor, MixTensor> {
    KleisliArrow::new(name, move |state: MixTensor| {
        let (ok, dissipation, violation) = check(&state);
        Admissible {
            value: state,
            result: AdmissibilityResult {
                admissible: ok,
                dissipation,
                violation,
            },
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monad_left_identity() {
        // Left identity: pure(a) >>= f  ≡  f(a)
        let val = 42.0_f32;
        let f = |x: f32| Admissible {
            value: x * 2.0,
            result: AdmissibilityResult {
                admissible: true,
                dissipation: 0.1,
                violation: None,
            },
        };

        let via_bind = Admissible::pure(val).bind(f);
        let direct = f(val);

        assert_eq!(via_bind.value, direct.value);
        assert_eq!(via_bind.result.admissible, direct.result.admissible);
    }

    #[test]
    fn test_monad_right_identity() {
        // Right identity: m >>= pure  ≡  m
        let m = Admissible {
            value: 42.0_f32,
            result: AdmissibilityResult {
                admissible: true,
                dissipation: 0.5,
                violation: None,
            },
        };
        let bound = m.clone().bind(Admissible::pure);
        assert_eq!(bound.value, m.value);
    }

    #[test]
    fn test_inadmissible_short_circuits() {
        let bad = Admissible {
            value: 1.0_f32,
            result: AdmissibilityResult {
                admissible: false,
                dissipation: -0.1,
                violation: Some("Clausius-Duhem".into()),
            },
        };

        let result = bad.bind(|x| Admissible::pure(x * 100.0));
        assert!(!result.result.admissible);
        assert_eq!(
            result.result.violation,
            Some("Clausius-Duhem".to_string())
        );
    }

    #[test]
    fn test_kleisli_arrow_basic() {
        let double = KleisliArrow::new("double", |x: f32| {
            Admissible::pure(x * 2.0)
        });

        let result = double.run(21.0);
        assert!(result.result.admissible);
        assert_eq!(result.value, 42.0);
    }

    #[test]
    fn test_gate_arrow_accepts() {
        let gate = gate_arrow("mass_conservation", |tensor: &MixTensor| {
            let total_mass: f32 = tensor.data().iter().take(1).sum();
            if total_mass > 0.0 {
                (true, 0.0, None)
            } else {
                (false, -1.0, Some("zero mass".into()))
            }
        });

        let mut t = MixTensor::new();
        t.add_material(350.0, 3.15, 0, 0.9, 100.0, 350.0, 2.8, 0.6);

        let result = gate.run(t);
        assert!(result.result.admissible);
    }
}

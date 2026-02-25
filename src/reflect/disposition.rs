//! Disposition engine — verbalizes bank personality into prompt components.

use crate::types::{BankPromptContext, MemoryBank};

/// Map a 1-5 scale value to a natural language description.
fn skepticism_label(level: u8) -> &'static str {
    match level {
        1 => "very trusting",
        2 => "somewhat trusting",
        3 => "balanced between trust and skepticism",
        4 => "quite skeptical",
        5 => "extremely skeptical",
        _ => unreachable!("disposition validated to 1-5"),
    }
}

fn literalism_label(level: u8) -> &'static str {
    match level {
        1 => "reads between the lines",
        2 => "mostly infers implied meaning",
        3 => "balances literal and implied meaning",
        4 => "interprets things fairly literally",
        5 => "interprets everything literally",
        _ => unreachable!("disposition validated to 1-5"),
    }
}

fn empathy_label(level: u8) -> &'static str {
    match level {
        1 => "focuses purely on logic",
        2 => "mostly focuses on logic with some consideration of feelings",
        3 => "balances logic and emotional factors",
        4 => "gives significant weight to emotional factors",
        5 => "strongly weights emotional factors",
        _ => unreachable!("disposition validated to 1-5"),
    }
}

fn bias_strength_label(strength: f32) -> &'static str {
    if strength < 0.25 {
        "minimally"
    } else if strength < 0.5 {
        "moderately"
    } else if strength < 0.75 {
        "significantly"
    } else {
        "strongly"
    }
}

/// Verbalize a memory bank's configuration into prompt components.
///
/// Pure function that maps disposition parameters to natural language and
/// formats directives and mission into prompt-ready strings.
pub fn verbalize_bank_profile(bank: &MemoryBank) -> BankPromptContext {
    let d = &bank.disposition;

    let disposition_prompt = format!(
        "Your reasoning style: You are {skepticism}. You {literalism}. You {empathy}. \
         These traits {bias} influence your responses.",
        skepticism = skepticism_label(d.skepticism()),
        literalism = literalism_label(d.literalism()),
        empathy = empathy_label(d.empathy()),
        bias = bias_strength_label(d.bias_strength()),
    );

    let directives_prompt = if bank.directives.is_empty() {
        String::new()
    } else {
        let rules: Vec<String> = bank
            .directives
            .iter()
            .enumerate()
            .map(|(i, d)| format!("{}. You MUST: {}", i + 1, d))
            .collect();
        format!("Hard rules (never violate):\n{}", rules.join("\n"))
    };

    let mission_prompt = if bank.mission.is_empty() {
        String::new()
    } else {
        bank.mission.clone()
    };

    BankPromptContext {
        disposition_prompt,
        directives_prompt,
        mission_prompt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BankId, Disposition};

    fn make_bank(skepticism: u8, literalism: u8, empathy: u8, bias_strength: f32) -> MemoryBank {
        MemoryBank {
            id: BankId::new(),
            name: "test".into(),
            mission: String::new(),
            directives: vec![],
            disposition: Disposition::new(skepticism, literalism, empathy, bias_strength).unwrap(),
        }
    }

    #[test]
    fn min_params() {
        let bank = make_bank(1, 1, 1, 0.0);
        let ctx = verbalize_bank_profile(&bank);
        assert!(ctx.disposition_prompt.contains("very trusting"));
        assert!(ctx.disposition_prompt.contains("reads between the lines"));
        assert!(ctx.disposition_prompt.contains("focuses purely on logic"));
        assert!(ctx.disposition_prompt.contains("minimally"));
    }

    #[test]
    fn max_params() {
        let bank = make_bank(5, 5, 5, 1.0);
        let ctx = verbalize_bank_profile(&bank);
        assert!(ctx.disposition_prompt.contains("extremely skeptical"));
        assert!(ctx.disposition_prompt.contains("interprets everything literally"));
        assert!(ctx.disposition_prompt.contains("strongly weights emotional factors"));
        assert!(ctx.disposition_prompt.contains("strongly influence"));
    }

    #[test]
    fn empty_directives() {
        let bank = make_bank(3, 3, 3, 0.5);
        let ctx = verbalize_bank_profile(&bank);
        assert!(ctx.directives_prompt.is_empty());
    }

    #[test]
    fn multiple_directives() {
        let mut bank = make_bank(3, 3, 3, 0.5);
        bank.directives = vec![
            "Never share private data".into(),
            "Always cite sources".into(),
            "Be concise".into(),
        ];
        let ctx = verbalize_bank_profile(&bank);
        assert!(ctx.directives_prompt.contains("1. You MUST: Never share private data"));
        assert!(ctx.directives_prompt.contains("2. You MUST: Always cite sources"));
        assert!(ctx.directives_prompt.contains("3. You MUST: Be concise"));
    }

    #[test]
    fn mission_verbatim() {
        let mut bank = make_bank(3, 3, 3, 0.5);
        bank.mission = "Remember developer context and preferences.".into();
        let ctx = verbalize_bank_profile(&bank);
        assert_eq!(
            ctx.mission_prompt,
            "Remember developer context and preferences."
        );
    }

    #[test]
    fn empty_mission() {
        let bank = make_bank(3, 3, 3, 0.5);
        let ctx = verbalize_bank_profile(&bank);
        assert!(ctx.mission_prompt.is_empty());
    }
}

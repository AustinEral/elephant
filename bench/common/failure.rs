use elephant::error::Error;

/// Returns whether a benchmark should abort immediately for this error.
pub fn is_fatal_bench_error(error: &Error) -> bool {
    match error {
        Error::InvalidDisposition(_)
        | Error::Serialization(_)
        | Error::InvalidId(_)
        | Error::Storage(_)
        | Error::NotFound(_)
        | Error::Embedding(_)
        | Error::Reranker(_)
        | Error::Internal(_)
        | Error::Configuration(_)
        | Error::EmbeddingDimensionMismatch { .. } => true,
        Error::Llm(message) => is_fatal_llm_error_message(message),
        Error::LlmNoJson | Error::LlmRefusal | Error::RateLimit(_) | Error::ServerError(_) => false,
    }
}

fn is_fatal_llm_error_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    if matches!(api_error_status_code(&lower), Some(status) if (400..500).contains(&status) && status != 429)
    {
        return true;
    }

    lower.contains("failed to parse openai response")
        || lower.contains("failed to parse anthropic response")
        || lower.contains("failed to parse gemini response")
}

fn api_error_status_code(message: &str) -> Option<u16> {
    let marker = "api error (";
    let start = message.find(marker)? + marker.len();
    let digits = message[start..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_400_is_fatal() {
        let err = Error::Llm(
            "OpenAI API error (400 Bad Request): Invalid schema for function 'search'".into(),
        );
        assert!(is_fatal_bench_error(&err));
    }

    #[test]
    fn rate_limit_is_not_fatal() {
        let err = Error::RateLimit("OpenAI API (429): slow down".into());
        assert!(!is_fatal_bench_error(&err));
    }

    #[test]
    fn llm_refusal_is_not_fatal() {
        assert!(!is_fatal_bench_error(&Error::LlmRefusal));
    }

    #[test]
    fn provider_response_parse_change_is_fatal() {
        let err = Error::Llm("failed to parse OpenAI response: missing field".into());
        assert!(is_fatal_bench_error(&err));
    }
}

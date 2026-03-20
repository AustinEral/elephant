//! Error-to-HTTP mapping for the API server.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use tracing::error;

use crate::error::Error;

/// JSON error body returned to clients.
#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = match &self {
            Error::NotFound(_) => StatusCode::NOT_FOUND,
            Error::InvalidDisposition(_)
            | Error::InvalidId(_)
            | Error::Serialization(_)
            | Error::Configuration(_) => {
                StatusCode::BAD_REQUEST
            }
            Error::Llm(_)
            | Error::LlmNoJson
            | Error::LlmRefusal
            | Error::Embedding(_)
            | Error::Reranker(_)
            | Error::ServerError(_) => StatusCode::BAD_GATEWAY,
            Error::RateLimit(_) => StatusCode::TOO_MANY_REQUESTS,
            Error::EmbeddingDimensionMismatch { .. } => StatusCode::CONFLICT,
            Error::Storage(_) | Error::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };

        let msg = self.to_string();
        error!(
            status = status.as_u16(),
            error = msg.as_str(),
            "request_error"
        );

        let body = ErrorBody { error: msg };
        (status, Json(body)).into_response()
    }
}

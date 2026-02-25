//! Error-to-HTTP mapping for the API server.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

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
            Error::InvalidDisposition(_) | Error::InvalidId(_) | Error::Serialization(_) => {
                StatusCode::BAD_REQUEST
            }
            Error::Llm(_) | Error::Embedding(_) => StatusCode::BAD_GATEWAY,
            Error::Storage(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };

        let body = ErrorBody {
            error: self.to_string(),
        };

        (status, Json(body)).into_response()
    }
}
